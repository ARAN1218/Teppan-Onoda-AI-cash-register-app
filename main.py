import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
import pytz
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import time # timeモジュールをインポート
from mlxtend.frequent_patterns import apriori, association_rules # 併売分析のために追加

# --- アプリケーションの基本設定 ---
st.set_page_config(
    page_title="鉄板おの田 AIレジ🤖",
    page_icon="🍳",
    layout="wide"
)

# --- 商品メニューと価格の定義 ---
MENU = {
    # 単品
    "焼きそば": 500,
    "焼きとうもろこし": 400,
    "フランクフルト": 300,
    "ラムネ": 250,
    "缶ジュース": 150,
    # セット
    "焼きそば&ラムネセット": 700,
    "焼きそば&缶ジュースセット": 600,
    # 割引セット
    "【経シス割引券】焼きそば&ラムネセット": 600,
    "【特別割引券】焼きそば&ラムネセット": 500,
    "【PiedPiper割引券】焼きそば&缶ジュースセット": 500,
}

# スプレッドシートのカラム順序を定義
SHEET_COLUMNS = [
    "タイムスタンプ", "TransactionID", "合計金額",
    "焼きそば", "焼きとうもろこし", "フランクフルト", "ラムネ", "缶ジュース",
    "焼きそば&ラムネセット", "焼きそば&缶ジュースセット",
    "【経シス割引券】焼きそば&ラムネセット", "【特別割引券】焼きそば&ラムネセット", "【PiedPiper割引券】焼きそば&缶ジュースセット"
]

# 日本時間のタイムゾーン
JST = pytz.timezone('Asia/Tokyo')

# --- セッションステートの初期化 ---
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'total_amount' not in st.session_state:
    st.session_state.total_amount = 0
if 'page' not in st.session_state:
    st.session_state.page = "register"

# --- Googleスプレッドシートへの接続とSecretsの検証 ---
secrets_ok = True
# 必須のキーが存在するかチェック
if 'gcp_service_account' not in st.secrets:
    st.error("`gcp_service_account` の情報が secrets.toml に設定されていません。")
    secrets_ok = False
if 'google_sheet_id' not in st.secrets:
    st.error("`google_sheet_id` が secrets.toml に設定されていません。")
    secrets_ok = False

# エラーがあればメッセージを表示して処理を停止
if not secrets_ok:
    st.warning("`README.md` の「ステップ3: StreamlitへのSecrets設定」を参考に、`.streamlit/secrets.toml` ファイルを正しく設定してください。")
    st.stop() # ここでアプリの実行を停止


@st.cache_resource
def get_gsheet_client():
    """サービスアカウント情報を使ってGoogleスプレッドシートに接続する"""
    try:
        # st.secretsから認証情報を辞書として取得
        creds = st.secrets["gcp_service_account"]
        # gspreadを使って認証
        gc = gspread.service_account_from_dict(creds)
        return gc
    except Exception as e:
        st.error(f"Googleスプレッドシートへの認証に失敗しました。: {e}")
        st.info("💡 `README.md` の指示に従って、サービスアカウントの設定とStreamlitのSecrets設定が正しく行われているか確認してください。")
        return None

# クライアントを取得
gc = get_gsheet_client()

# --- データ読み書き関数 ---
@st.cache_data(ttl=60) # 60秒間キャッシュ
def load_data_from_sheet(_gc):
    """スプレッドシートからデータを読み込む"""
    if _gc is None:
        return pd.DataFrame(columns=SHEET_COLUMNS)
    try:
        spreadsheet = _gc.open_by_key(st.secrets["google_sheet_id"])
        worksheet = spreadsheet.worksheet("売上データ")
        df = get_as_dataframe(worksheet, header=0, usecols=list(range(len(SHEET_COLUMNS))))
        df.dropna(how='all', inplace=True) # 全てが空の行を削除
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("スプレッドシートが見つかりません。Secretsの`google_sheet_id`が正しいか、サービスアカウントにシートが共有されているか確認してください。")
        return pd.DataFrame(columns=SHEET_COLUMNS)
    except gspread.exceptions.WorksheetNotFound:
        st.error("スプレッドシート内に「売上データ」という名前のシート（タブ）が見つかりません。シート名を確認してください。")
        return pd.DataFrame(columns=SHEET_COLUMNS)
    except Exception as e:
        st.error(f"データの読み込み中に予期せぬエラーが発生しました: {e}")
        return pd.DataFrame(columns=SHEET_COLUMNS)

# --- ヘルパー関数 ---
def add_to_cart(item_name):
    """カートに商品を追加し、フィードバックを表示する"""
    st.session_state.cart.append(item_name)
    update_total()
    st.toast(f'「{item_name}」をカートに追加しました！', icon='👍')


def update_total():
    """カート内の合計金額を計算して更新する"""
    st.session_state.total_amount = sum(MENU[item] for item in st.session_state.cart)

def clear_cart():
    """カートを空にする"""
    st.session_state.cart = []
    st.session_state.total_amount = 0
    st.session_state.page = "register"

def format_cart_df():
    """カート内の商品をDataFrame形式で整形する"""
    if not st.session_state.cart:
        return pd.DataFrame({"商品": [], "価格": [], "数量": []})
    
    # 商品ごとの数量をカウント
    item_counts = pd.Series(st.session_state.cart).value_counts().reset_index()
    item_counts.columns = ['商品', '数量']
    # 価格情報をマージ
    item_counts['価格'] = item_counts['商品'].map(MENU)
    
    return item_counts[['商品', '価格', '数量']]


# --- データ分析タブで使う関数 ---
def preprocess_data(df):
    """データ分析のための前処理を行う"""
    if df.empty:
        return df
    
    df_processed = df.copy()
    
    # データ型の変換
    # `errors='coerce'` は変換できない値を `NaT` (Not a Time) にします
    df_processed["タイムスタンプ"] = pd.to_datetime(df_processed["タイムスタンプ"], errors='coerce')
    # 不正な日付データを削除
    df_processed.dropna(subset=["タイムスタンプ"], inplace=True)
    
    df_processed["合計金額"] = pd.to_numeric(df_processed["合計金額"], errors='coerce')
    
    # 時間帯データを追加
    df_processed['時間帯'] = df_processed['タイムスタンプ'].dt.hour
    
    # 商品カラムを数値型に変換
    for col in SHEET_COLUMNS[3:]:
        # カラムが存在するか確認してから処理する
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        
    return df_processed


# --- UI描画 ---
# タブを作成
tab1, tab2 = st.tabs(["🛒 レジ", "📊 データ分析"])

# --- レジタブのUI ---
with tab1:
    if st.session_state.page == "register":
        st.title("🍳 鉄板おの田 AIレジ")
        
        # 2カラムレイアウトに戻す
        col1, col2 = st.columns([2, 1])
        
        # col1: メニューボタン
        with col1:
            st.header("メニュー")
            
            # 商品カテゴリごとにボタンを配置
            st.subheader("フード")
            food_cols = st.columns(3)
            with food_cols[0]:
                if st.button("焼きそば (¥500)", use_container_width=True):
                    add_to_cart("焼きそば")
            with food_cols[1]:
                if st.button("焼きとうもろこし (¥400)", use_container_width=True):
                    add_to_cart("焼きとうもろこし")
            with food_cols[2]:
                if st.button("フランクフルト (¥300)", use_container_width=True):
                    add_to_cart("フランクフルト")
                    
            st.subheader("ドリンク")
            drink_cols = st.columns(3)
            with drink_cols[0]:
                if st.button("ラムネ (¥250)", use_container_width=True):
                    add_to_cart("ラムネ")
            with drink_cols[1]:
                if st.button("缶ジュース (¥150)", use_container_width=True):
                    add_to_cart("缶ジュース")
            
            st.subheader("セットメニュー")
            set_cols = st.columns(2)
            with set_cols[0]:
                if st.button("焼きそば&ラムネセット (¥700)", use_container_width=True):
                    add_to_cart("焼きそば&ラムネセット")
            with set_cols[1]:
                if st.button("焼きそば&缶ジュースセット (¥600)", use_container_width=True):
                    add_to_cart("焼きそば&缶ジュースセット")

            st.subheader("割引券セット")
            discount_cols = st.columns(3)
            with discount_cols[0]:
                 if st.button("【経シス割引券】焼きそば&ラムネセット (¥600)", use_container_width=True):
                    add_to_cart("【経シス割引券】焼きそば&ラムネセット")
            with discount_cols[1]:
                 if st.button("【特別割引券】焼きそば&ラムネセット (¥500)", use_container_width=True):
                    add_to_cart("【特別割引券】焼きそば&ラムネセット")
            with discount_cols[2]:
                 if st.button("【PiedPiper割引券】焼きそば&缶ジュースセット (¥500)", use_container_width=True):
                    add_to_cart("【PiedPiper割引券】焼きそば&缶ジュースセット")
        
        # col2: 注文内容と確定ボタン
        with col2:
            st.header("現在の注文")
            
            if not st.session_state.cart:
                st.info("商品ボタンを押して注文を追加してください。")
            else:
                # カートの中身を整形して表示
                cart_df = format_cart_df()
                st.dataframe(cart_df, hide_index=True, use_container_width=True)
                
                # 合計金額
                st.metric(label="お会計", value=f"¥ {st.session_state.total_amount:,}")
                
                # 操作ボタン
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("注文を確定", type="primary", use_container_width=True, disabled=(gc is None)):
                        st.session_state.page = "confirm"
                        st.rerun() # ページを再読み込みして表示を切り替え
                with btn_cols[1]:
                    if st.button("クリア", use_container_width=True):
                        clear_cart()
                        st.rerun()


    elif st.session_state.page == "confirm":
        st.title("お会計確認")
        
        cart_df = format_cart_df()
        st.dataframe(cart_df, hide_index=True, use_container_width=True)
        
        st.markdown(f"""
        <div style="text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
            <h2 style="color: #333;">合計金額</h2>
            <p style="font-size: 48px; font-weight: bold; color: #1E90FF; margin: 0;">
                ¥ {st.session_state.total_amount:,}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("お客様に合計金額をお伝えし、代金を受け取ってください。")
        
        # 会計完了とキャンセルボタン
        btn_cols = st.columns(2)
        with btn_cols[0]:
            if st.button("✅ 会計完了", type="primary", help="このボタンを押すと売上が記録されます", use_container_width=True):
                if gc:
                    # スプレッドシートに書き込むデータを作成
                    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
                    transaction_id = str(uuid.uuid4())
                    total = st.session_state.total_amount
                    
                    # 商品ごとの数量をカウント
                    item_counts = {item: 0 for item in MENU.keys()}
                    for item in st.session_state.cart:
                        item_counts[item] += 1
                    
                    # スプレッドシートのカラム順に従ってデータを作成
                    new_row_data = [now, transaction_id, total] + [item_counts.get(col, 0) for col in SHEET_COLUMNS[3:]]
                    
                    try:
                        # データを1行追記する
                        spreadsheet = gc.open_by_key(st.secrets["google_sheet_id"])
                        worksheet = spreadsheet.worksheet("売上データ")
                        worksheet.append_row(new_row_data, value_input_option='USER_ENTERED')
                        
                        st.success("売上を記録しました！ありがとうございました！")
                        st.balloons()
                        
                        # データ分析タブのキャッシュをクリアして最新情報を反映
                        st.cache_data.clear()
                        
                        time.sleep(2) # 2秒待機
                        
                        clear_cart()
                        st.session_state.page = "register" # レジ画面に戻る
                        st.rerun()
                    except gspread.exceptions.WorksheetNotFound:
                        st.error("データの書き込みに失敗しました。Googleスプレッドシートに「売上データ」という名前のシート（タブ）が存在するか確認してください。")
                        st.info("シート名が異なっている場合（例：「シート1」）、正しい名前に変更してください。")
                    except Exception as e:
                        st.error(f"予期せぬエラーでデータの書き込みに失敗しました。: {e}")

                else:
                    st.error("Googleスプレッドシートに接続できません。設定を確認してください。")

        with btn_cols[1]:
            if st.button("🔙 修正する", use_container_width=True):
                st.session_state.page = "register"
                st.rerun()

# --- データ分析タブのUI ---
with tab2:
    st.title("📊 リアルタイム売上分析")
    
    if not gc:
        st.error("Googleスプレッドシートに接続できていないため、分析データを表示できません。")
    else:
        # データの読み込みと前処理
        df_raw = load_data_from_sheet(gc)

        if df_raw.empty:
            st.warning("まだ売上データがありません。会計が完了すると、ここに分析結果が表示されます。")
        else:
            df = preprocess_data(df_raw)

            # --- 分析用データの前処理 (セットメニューの集約) ---
            df_analysis = df.copy()

            # 割引セットを通常セットに合算
            df_analysis['焼きそば&ラムネセット'] = df[['焼きそば&ラムネセット', '【経シス割引券】焼きそば&ラムネセット', '【特別割引券】焼きそば&ラムネセット']].sum(axis=1)
            df_analysis['焼きそば&缶ジュースセット'] = df[['焼きそば&缶ジュースセット', '【PiedPiper割引券】焼きそば&缶ジュースセット']].sum(axis=1)

            # 分析で使う商品カラムリスト（割引セットは除く）
            product_cols_for_analysis = [
                "焼きそば", "焼きとうもろこし", "フランクフルト", "ラムネ", "缶ジュース",
                "焼きそば&ラムネセット", "焼きそば&缶ジュースセット"
            ]

            # 1. サマリー
            st.header("📈 サマリー")
            total_sales = df['合計金額'].sum()
            total_transactions = len(df)
            avg_sales_per_customer = total_sales / total_transactions if total_transactions > 0 else 0
            
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("総売上高", f"¥ {total_sales:,.0f}")
            with summary_cols[1]:
                st.metric("総販売件数 (会計回数)", f"{total_transactions} 件")
            with summary_cols[2]:
                st.metric("平均客単価", f"¥ {avg_sales_per_customer:,.0f}")

            st.divider()

            # 2. 商品別分析
            st.header("🍔 商品別分析")
            
            if not df_analysis.empty and all(col in df_analysis.columns for col in product_cols_for_analysis):
                quantities = df_analysis[product_cols_for_analysis].sum()
                
                # 価格のSeriesを作成（集約後の商品リストで）
                prices_for_analysis = {k: v for k, v in MENU.items() if k in product_cols_for_analysis}
                prices = pd.Series(prices_for_analysis)[quantities.index]
                
                sales_by_product = quantities * prices
                product_sales = pd.DataFrame({
                    '販売数量': quantities,
                    '売上金額': sales_by_product
                }).reset_index().rename(columns={'index': '商品'})

            else:
                product_sales = pd.DataFrame(columns=['商品', '販売数量', '売上金額'])

            # ランキング表示
            col_rank1, col_rank2 = st.columns(2)
            with col_rank1:
                st.subheader("💰 売上金額ランキング")
                top_sales = product_sales.sort_values('売上金額', ascending=False).reset_index(drop=True)
                st.dataframe(top_sales.style.background_gradient(subset=['売上金額'], cmap='Reds'), hide_index=True, use_container_width=True)
            with col_rank2:
                st.subheader("🔢 販売数量ランキング")
                top_quantity = product_sales.sort_values('販売数量', ascending=False).reset_index(drop=True)
                st.dataframe(top_quantity.style.background_gradient(subset=['販売数量'], cmap='Blues'), hide_index=True, use_container_width=True)

            # 売上構成比 (円グラフ)
            st.subheader("🍰 売上構成比")
            # データを売上金額でソート
            sorted_product_sales = product_sales.sort_values('売上金額', ascending=False)
            fig_pie = px.pie(sorted_product_sales[sorted_product_sales['売上金額']>0], names='商品', values='売上金額', 
                             title='商品別の売上構成比', hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', sort=False) # Plotly側でのソートは無効化
            st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            # 3. 時間帯別分析
            st.header("🕒 時間帯別分析")
            time_interval = st.radio("集計間隔を選択（分）", [10, 20, 30, 60], horizontal=True, index=3)
            
            # タイムスタンプをインデックスに設定
            df_time_analysis = df.set_index('タイムスタンプ')
            
            # 指定した間隔でリサンプリング（集計）
            time_binned = df_time_analysis.resample(f'{time_interval}T').agg(
                販売件数=('TransactionID', 'count'),
                売上=('合計金額', 'sum')
            ).reset_index()
            
            fig_hist = px.bar(time_binned, x='タイムスタンプ', y='販売件数', title=f'{time_interval}分間の販売件数推移',
                              hover_data=['売上'])
            fig_hist.update_xaxes(title_text='時間')
            fig_hist.update_yaxes(title_text='販売件数')
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.divider()

            # 4. 併売分析 (アソシエーション分析)
            st.header("🤝 併売分析 (アソシエーションルール)")
            st.info("""
            **支持度 (Support):** 全体の中で、商品AとBが同時に買われる確率。
            **信頼度 (Confidence):** 商品Aを買った人が、商品Bも買う確率。
            **リフト値 (Lift):** 商品B単体で売れる確率に比べ、Aを買ったことでBが売れる確率が何倍になったか。**1より大きいと正の相関**があり、値が大きいほど関連性が強いとされます。
            """)

            # 分析用のデータ（購入したかどうかをTrue/Falseで表現）
            basket_sets = df_analysis[product_cols_for_analysis] > 0
            
            if len(basket_sets) > 10: # データが少ないと分析できないため
                # 支持度が高い商品ペアを抽出 (min_supportはデータ量に応じて調整)
                frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
                
                if not frequent_itemsets.empty:
                    # アソシエーションルールを計算
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                    
                    if not rules.empty:
                        # 結果の整形
                        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        st.subheader("📈 リフト値TOP10の組み合わせ")
                        display_rules = rules.sort_values('lift', ascending=False).head(10)
                        st.dataframe(display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']], hide_index=True, use_container_width=True)
                    else:
                        st.warning("リフト値が1を超える意味のある組み合わせは見つかりませんでした。")
                else:
                    st.warning("頻繁に購入される商品の組み合わせが見つかりませんでした。")
            else:
                st.warning("分析するには、あと " + str(11 - len(basket_sets)) + " 件以上の取引データが必要です。")


            st.divider()

            # 5 & 6. セットメニューと割引券の効果測定 (元のdfを使用)
            st.header("🎁 セットメニュー・割引券の効果測定")
            
            set_menu_data = {
                'メニュー': [
                    '焼きそば&ラムネセット', '焼きそば&缶ジュースセット',
                    '【経シス割引券】焼きそば&ラムネセット', '【特別割引券】焼きそば&ラムネセット',
                    '【PiedPiper割引券】焼きそば&缶ジュースセット'
                ],
                '販売数': [df[name].sum() if name in df else 0 for name in [
                    '焼きそば&ラムネセット', '焼きそば&缶ジュースセット',
                    '【経シス割引券】焼きそば&ラムネセット', '【特別割引券】焼きそば&ラムネセット',
                    '【PiedPiper割引券】焼きそば&缶ジュースセット'
                ]]
            }
            set_menu_df = pd.DataFrame(set_menu_data)

            # 割引券の利用率
            total_sets = set_menu_df['販売数'].sum()
            discount_sets = set_menu_df[set_menu_df['メニュー'].str.contains('割引券')]['販売数'].sum()
            discount_rate = (discount_sets / total_sets * 100) if total_sets > 0 else 0
            
            set_cols = st.columns(2)
            with set_cols[0]:
                st.subheader("セットメニュー販売数")
                st.dataframe(set_menu_df, hide_index=True, use_container_width=True)
            with set_cols[1]:
                st.subheader("割引券利用状況")
                st.metric("全セット販売数", f"{total_sets:.0f} 個")
                st.metric("うち割引券利用数", f"{discount_sets:.0f} 個")
                st.metric("割引券利用率", f"{discount_rate:.1f} %")


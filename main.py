import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
import pytz
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe

# --- アプリケーションの基本設定 ---
st.set_page_config(
    page_title="学園祭 鉄板焼きレジシステム",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
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
    "焼きそば&ラムネセット", "焼きそば&缶ジュースセット", # "ラム네セット" のタイポを修正
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
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        
    return df_processed


# --- UI描画 ---
# タブを作成
tab1, tab2 = st.tabs(["🛒 レジ", "📊 データ分析"])


# --- レジタブのUI ---
with tab1:
    if st.session_state.page == "register":
        st.title("🍳 鉄板焼き レジシステム")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("メニュー")
            
            # 商品カテゴリごとにボタンを配置
            st.subheader("フード")
            food_cols = st.columns(3)
            with food_cols[0]:
                if st.button("焼きそば (¥500)", use_container_width=True):
                    st.session_state.cart.append("焼きそば")
                    update_total()
            with food_cols[1]:
                if st.button("焼きとうもろこし (¥400)", use_container_width=True):
                    st.session_state.cart.append("焼きとうもろこし")
                    update_total()
            with food_cols[2]:
                if st.button("フランクフルト (¥300)", use_container_width=True):
                    st.session_state.cart.append("フランクフルト")
                    update_total()
                    
            st.subheader("ドリンク")
            drink_cols = st.columns(3)
            with drink_cols[0]:
                if st.button("ラムネ (¥250)", use_container_width=True):
                    st.session_state.cart.append("ラムネ")
                    update_total()
            with drink_cols[1]:
                if st.button("缶ジュース (¥150)", use_container_width=True):
                    st.session_state.cart.append("缶ジュース")
                    update_total()
            
            st.subheader("セットメニュー")
            set_cols = st.columns(2)
            with set_cols[0]:
                if st.button("焼きそば&ラムネセット (¥700)", use_container_width=True):
                    st.session_state.cart.append("焼きそば&ラムネセット")
                    update_total()
            with set_cols[1]:
                if st.button("焼きそば&缶ジュースセット (¥600)", use_container_width=True):
                    st.session_state.cart.append("焼きそば&缶ジュースセット")
                    update_total()

            st.subheader("割引券セット")
            discount_cols = st.columns(3)
            with discount_cols[0]:
                 if st.button("【経シス割引券】焼きそば&ラムネセット (¥600)", use_container_width=True):
                    st.session_state.cart.append("【経シス割引券】焼きそば&ラムネセット")
                    update_total()
            with discount_cols[1]:
                 if st.button("【特別割引券】焼きそば&ラムネセット (¥500)", use_container_width=True):
                    st.session_state.cart.append("【特別割引券】焼きそば&ラムネセット")
                    update_total()
            with discount_cols[2]:
                 if st.button("【PiedPiper割引券】焼きそば&缶ジュースセット (¥500)", use_container_width=True):
                    st.session_state.cart.append("【PiedPiper割引券】焼きそば&缶ジュースセット")
                    update_total()


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
                        
                        st.success("売上を記録しました！")
                        st.balloons()
                        
                        # データ分析タブのキャッシュをクリアして最新情報を反映
                        st.cache_data.clear()
                        
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
            
            # 商品カラムのみを抽出
            product_cols = [col for col in SHEET_COLUMNS if col in MENU]
            
            # --- 修正箇所 ---
            # 商品ごとの販売数量と売上を計算（ベクトル化による効率化とバグ修正）
            # この修正は、お手元のファイルで発生している KeyError の原因である、
            # 誤った計算ロジックを、より安全で効率的な方法に置き換えるものです。
            if not df.empty and all(col in df.columns for col in product_cols):
                # 商品ごとの販売数量を合計
                quantities = df[product_cols].sum()
                
                # 商品名に対応する価格のSeriesを作成
                prices = pd.Series(MENU)[quantities.index]
                
                # 販売数量と価格を要素ごとに掛けて売上を算出
                sales_by_product = quantities * prices
                
                # 分析用のDataFrameを作成
                product_sales = pd.DataFrame({
                    '販売数量': quantities,
                    '売上金額': sales_by_product
                }).reset_index().rename(columns={'index': '商品'})

            else:
                # データがない場合の空のDataFrame
                product_sales = pd.DataFrame(columns=['商品', '販売数量', '売上金額'])
            # --- 修正ここまで ---

            # ランキング表示
            col_rank1, col_rank2 = st.columns(2)
            with col_rank1:
                st.subheader("💰 売上金額ランキング")
                top_sales = product_sales.sort_values('売上金額', ascending=False).reset_index(drop=True)
                st.dataframe(top_sales, hide_index=True, use_container_width=True)
            with col_rank2:
                st.subheader("🔢 販売数量ランキング")
                top_quantity = product_sales.sort_values('販売数量', ascending=False).reset_index(drop=True)
                st.dataframe(top_quantity, hide_index=True, use_container_width=True)

            # 売上構成比 (円グラフ)
            st.subheader("🍰 売上構成比")
            fig_pie = px.pie(product_sales[product_sales['売上金額']>0], names='商品', values='売上金額', 
                             title='商品別の売上構成比', hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            # 3. 時間帯別分析
            st.header("🕒 時間帯別分析")
            hourly_analysis = df.groupby('時間帯').agg(
                売上=('合計金額', 'sum'),
                販売件数=('TransactionID', 'count')
            ).reset_index()
            
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Scatter(x=hourly_analysis['時間帯'], y=hourly_analysis['売上'],
                                          mode='lines+markers', name='売上', yaxis='y1'))
            fig_hourly.add_trace(go.Scatter(x=hourly_analysis['時間帯'], y=hourly_analysis['販売件数'],
                                          mode='lines+markers', name='販売件数', yaxis='y2'))

            fig_hourly.update_layout(
                title="時間帯別の売上・販売件数 推移",
                xaxis_title="時間帯",
                yaxis_title="売上 (円)",
                yaxis2=dict(title="販売件数", overlaying='y', side='right'),
                legend=dict(x=0.1, y=0.9)
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

            st.divider()

            # 4. 併売分析
            st.header("🤝 併売分析")
            st.write("「この商品を買った人は、こちらも買っています」分析")
            
            # 分析対象の商品を選択
            target_product = st.selectbox("分析したい商品を選択してください", options=product_cols)
            
            if target_product:
                # 対象商品を購入したトランザクションを抽出
                bought_together_df = df[df[target_product] > 0]
                
                if not bought_together_df.empty:
                    # 対象商品以外の商品の購入数を集計
                    other_products = bought_together_df[product_cols].drop(columns=[target_product]).sum()
                    other_products = other_products[other_products > 0].sort_values(ascending=False).reset_index()
                    other_products.columns = ['一緒に買われた商品', '購入数']
                    
                    if not other_products.empty:
                        st.write(f"**「{target_product}」** と一緒に買われている商品:")
                        fig_basket = px.bar(other_products, x='一緒に買われた商品', y='購入数',
                                            text_auto=True, title=f"「{target_product}」との併売ランキング")
                        st.plotly_chart(fig_basket, use_container_width=True)
                    else:
                        st.info(f"「{target_product}」はまだ他の商品と一緒には購入されていません。")
                else:
                    st.info(f"「{target_product}」はまだ購入されていません。")

            st.divider()

            # 5 & 6. セットメニューと割引券の効果測定
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



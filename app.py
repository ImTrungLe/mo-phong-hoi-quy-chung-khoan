import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# Cấu hình trang
st.set_page_config(page_title="Mô phỏng Hồi quy Chứng khoán", layout="wide")
st.title("📊 Ứng Dụng Mô Phỏng & Thực Nghiệm Các Mô Hình Hồi Quy Tài Chính")

# Chia các tab nội dung theo đúng cấu trúc bài thuyết trình
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Hồi quy tuyến tính tĩnh", 
    "2. Mô hình tự hồi quy AR(1)", 
    "3. Tiến trình liên tục O-U",
    "4. 📈 Thực nghiệm dữ liệu thật"
])

# ==========================================
# TAB 1: HỒI QUY TUYẾN TÍNH TĨNH
# ==========================================
with tab1:
    st.header("Hồi quy tuyến tính tĩnh (Linear Regression)")
    st.latex(r"Y = a + bX")
    st.write("Sử dụng phương pháp Bình phương tối thiểu (OLS) để vẽ một đường thẳng xu hướng xuyên qua các điểm giá.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        n_points = st.slider("Số phiên giao dịch (N)", 50, 500, 200, step=50, key="t1_n")
        slope = st.slider("Hệ số góc b (Độ dốc xu hướng)", -0.5, 0.5, 0.1, step=0.05, key="t1_b")
        intercept = st.slider("Hệ số chặn a", 10, 100, 50, key="t1_a")
        noise_level = st.slider("Độ nhiễu thị trường", 1, 20, 5, key="t1_noise")
        
    with col2:
        X = np.arange(n_points)
        pure_Y = intercept + slope * X
        noise = np.random.normal(0, noise_level, n_points)
        Y = pure_Y + noise
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='🔵 Điểm giá thực tế', marker=dict(color='blue', opacity=0.5)))
        fig1.add_trace(go.Scatter(x=X, y=pure_Y, mode='lines', name='🔴 Đường xu hướng (OLS)', line=dict(color='red', width=3)))
        
        # Thêm nhãn giải thích trực tiếp lên đồ thị
        fig1.add_annotation(x=n_points/2, y=pure_Y[int(n_points/2)] + noise_level*2, 
                            text="Khoảng cách từ điểm xanh đến đường đỏ chính là Sai số (Residuals)", 
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black")
        
        fig1.update_layout(title="Mô phỏng đường xu hướng tĩnh", xaxis_title="Thời gian (Phiên)", yaxis_title="Giá cổ phiếu", height=450, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig1, use_container_width=True)

# ==========================================
# TAB 2: MÔ HÌNH TỰ HỒI QUY RỜI RẠC AR(1)
# ==========================================
with tab2:
    st.header("Mô hình tự hồi quy rời rạc AR(1)")
    st.latex(r"X_t = \alpha + \beta X_{t-1} + \epsilon_t")
    st.write("Mô hình thừa nhận tính phụ thuộc tuyến tính mang bản chất Markovian: giá trị tài sản ở hiện tại chịu tác động trực tiếp từ giá trị liền kề trước đó của chính nó.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        alpha = st.slider("Hệ số tự do alpha", 1, 10, 5, key="t2_alpha")
        beta = st.slider("Hệ số tự hồi quy beta", -1.5, 1.5, 0.7, step=0.05, key="t2_beta")
        st.write("💡 *Điều kiện tồn tại trung bình hồi quy: |beta| < 1. Nếu |beta| >= 1, chuỗi sẽ rơi vào trạng thái bước ngẫu nhiên hoặc bùng nổ.*")
        
    with col2:
        np.random.seed(42)
        ar_data = np.zeros(200)
        ar_data[0] = 50
        for t in range(1, 200):
            ar_data[t] = alpha + beta * ar_data[t-1] + np.random.normal(0, 2)
            if abs(ar_data[t]) > 10000:
                ar_data[t:] = np.nan
                break
                
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=ar_data, mode='lines', name='🔵 Quỹ đạo giá AR(1)', line=dict(color='blue')))
        if abs(beta) < 1:
            mean_long_term = alpha / (1 - beta)
            fig2.add_shape(type="line", x0=0, y0=mean_long_term, x1=200, y1=mean_long_term, line=dict(color="Green", width=3, dash="dash"), name="Mức cân bằng")
            
            # Thêm annotation nhãn mác
            fig2.add_annotation(x=100, y=mean_long_term, text="🟢 Lực Nam châm (Trục cân bằng dài hạn)", showarrow=True, arrowhead=2, ax=0, ay=-40)
            
            st.success(f"Chuỗi đạt trạng thái dừng. Trục trung bình hồi quy dài hạn: {mean_long_term:.2f}")
        else:
            st.error("Chuỗi không dừng (Bùng nổ hoặc Bước ngẫu nhiên)! Không tồn tại xu hướng hội tụ về mức cân bằng.")
            
        fig2.update_layout(title="Mô phỏng tính dừng và hội tụ của mô hình rời rạc", xaxis_title="Thời gian (t)", yaxis_title="Giá trị Xt", height=450)
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 3: TIẾN TRÌNH NGẪU NHIÊN LIÊN TỤC O-U
# ==========================================
with tab3:
    st.header("Tiến trình ngẫu nhiên liên tục Ornstein - Uhlenbeck (O-U Process)")
    st.latex(r"dX_t = \theta(\mu - X_t)dt + \sigma dW_t")
    st.write("Mô hình hóa dòng chảy liên tục của thị trường, mô tả chính xác lực kéo hồi quy về mức cân bằng dài hạn lý thuyết.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        theta = st.slider("Hệ số tốc độ hồi quy (theta)", 0.05, 1.00, 0.30, step=0.05, key="t3_theta")
        mu = st.slider("Mức trung bình dài hạn (mu)", 50, 150, 100, key="t3_mu")
        sigma = st.slider("Hệ số biến động / Độ rủi ro (sigma)", 1, 20, 5, key="t3_sigma")
        
    with col2:
        T = 10.0
        N = 500
        dt = T / N
        t_space = np.linspace(0, T, N)
        
        X_ou = np.zeros(N)
        X_ou[0] = 60 
        
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            X_ou[i] = X_ou[i-1] + theta * (mu - X_ou[i-1]) * dt + sigma * dW
            
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t_space, y=X_ou, mode='lines', name='🔵 Quỹ đạo giá liên tục', line=dict(color='blue')))
        fig3.add_shape(type="line", x0=0, y0=mu, x1=T, y1=mu, line=dict(color="orange", width=3, dash="dash"), name="Mức trung bình mu")
        
        # Thêm annotation giải thích cơ chế
        fig3.add_annotation(x=T/2, y=mu, text="🟠 Trục cân bằng lý thuyết", showarrow=True, ax=0, ay=-40)
        fig3.add_annotation(x=T/4, y=np.max(X_ou), text="Lực hút tỷ lệ thuận với khoảng cách (Vận tốc θ)", showarrow=False, font=dict(color="red"))

        fig3.update_layout(title="Mô phỏng lực kéo liên tục của tiến trình O-U", xaxis_title="Thời gian liên tục (t)", yaxis_title="Giá tài sản Xt", height=450, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# TAB 4: THỰC NGHIỆM ĐỊNH LƯỢNG & MACHINE LEARNING
# ==========================================
with tab4:
    st.header("⚡ Tích hợp Machine Learning Khuyến nghị Giao dịch")
    st.write("Hệ thống tự động dùng Toán học trích xuất đặc trưng (Feature Engineering) và huấn luyện mô hình Logistic Regression để đưa ra tín hiệu MUA/BÁN cho ngày hôm nay.")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Nhập mã cổ phiếu (Yahoo Finance):", "AAPL")
        st.caption("Ví dụ: AAPL, MSFT, VNM.VN, FPT.VN")
        
        days_lookback = st.slider("Dữ liệu quá khứ (để train bot)", 200, 1000, 500)
        run_btn = st.button("Chạy Bot Định Lượng")
        
    with col2:
        if run_btn or ticker:
            try:
                # 1. Tải dữ liệu
                end_date = datetime.today()
                start_date = end_date - timedelta(days=days_lookback)
                
                with st.spinner('Đang tải data và huấn luyện mô hình...'):
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    st.error("Không tìm thấy dữ liệu.")
                else:
                    # Ép kiểu dữ liệu đa tầng của yfinance về dạng Series 1D đơn giản
                    close_series = data['Close'].squeeze()
                    df = pd.DataFrame({'Close': close_series})
                    
                    # 2. FEATURE ENGINEERING (Dùng Toán học tạo đặc trưng)
                    # Tính lợi nhuận hằng ngày (Momentum)
                    df['Return'] = df['Close'].pct_change()
                    
                    # Tính Trục trung bình và Độ lệch chuẩn trượt (Rolling 20 ngày)
                    window = 20
                    df['Rolling_Mean'] = df['Close'].rolling(window=window).mean()
                    df['Rolling_Std'] = df['Close'].rolling(window=window).std()
                    
                    # Trích xuất Z-Score: Đo lường khoảng cách từ giá đến trục cân bằng
                    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']
                    
                    # 3. LABELING (Gán nhãn để huấn luyện)
                    # Nhìn về tương lai 5 ngày (1 tuần giao dịch)
                    df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1
                    
                    # Quy tắc: Tương lai tăng > 2% là MUA (1), giảm < -2% là BÁN (-1), còn lại ĐỨNG NGOÀI (0)
                    df['Label'] = np.where(df['Future_Return'] > 0.02, 1, 
                                           np.where(df['Future_Return'] < -0.02, -1, 0))
                    
                    # Xóa các dòng NaN do shift và rolling gây ra
                    df_clean = df.dropna()
                    
                    # 4. TRAINING MACHINE LEARNING
                    X = df_clean[['Z_Score', 'Return']]
                    y = df_clean['Label']
                    
                    # Chuẩn hóa dữ liệu
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Dùng Logistic Regression với class_weight='balanced' để trị dữ liệu mất cân bằng
                    model = LogisticRegression(class_weight='balanced', random_state=42)
                    model.fit(X_scaled, y)
                    
                    # 5. PREDICTION (Dự báo cho ngày HIỆN TẠI)
                    # Lấy dữ liệu của ngày mới nhất (dòng cuối cùng)
                    latest_z = (df['Close'].iloc[-1] - df['Close'].rolling(window=window).mean().iloc[-1]) / df['Close'].rolling(window=window).std().iloc[-1]
                    latest_ret = df['Return'].iloc[-1]
                    
                    latest_features = scaler.transform([[latest_z, latest_ret]])
                    prediction = model.predict(latest_features)[0]
                    prob = model.predict_proba(latest_features)[0]
                    
                    # Xử lý UI hiển thị tín hiệu
                    st.subheader("🤖 Tín hiệu Bot Học Máy (Dự phóng 5 ngày tới)")
                    
                    if prediction == 1:
                        st.success(f"🔥 KHUYẾN NGHỊ: **MUA (BUY)**")
                        st.write(f"Độ tự tin của mô hình: {prob[2]*100:.1f}%")
                        st.info("💡 Giải thích: Z-Score cho thấy giá đang bị ép xuống vùng quá bán. Mô hình nhận diện được mẫu hình tương đồng trong quá khứ thường dẫn đến nhịp bật tăng phục hồi.")
                    elif prediction == -1:
                        st.error(f"❄️ KHUYẾN NGHỊ: **BÁN (SELL)**")
                        st.write(f"Độ tự tin của mô hình: {prob[0]*100:.1f}%")
                        st.info("💡 Giải thích: Sợi dây thun giá đang kéo quá căng (Z-Score cao). Dòng tiền hưng phấn có dấu hiệu đạt đỉnh, lực hồi quy chuẩn bị kéo giá rơi xuống.")
                    else:
                        st.warning(f"⚖️ KHUYẾN NGHỊ: **ĐỨNG NGOÀI (HOLD)**")
                        st.write(f"Độ tự tin của mô hình: {prob[1]*100:.1f}%")
                        st.info("💡 Giải thích: Các đặc trưng toán học đang ở trạng thái nhiễu. Lực mua và bán cân bằng. Vào lệnh lúc này rủi ro cao.")
                        
                    # Vẽ đồ thị minh họa lịch sử giá và Z-Score
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Giá thực tế"))
                    fig4.add_trace(go.Scatter(x=df.index, y=df['Rolling_Mean'], name="Trục cân bằng (20 ngày)", line=dict(dash='dash', color='orange')))
                    fig4.update_layout(title=f"Đồ thị vận động giá {ticker}", xaxis_title="Thời gian", yaxis_title="Giá", height=400)
                    st.plotly_chart(fig4, use_container_width=True)

            except Exception as e:
                st.error(f"Đã xảy ra lỗi hệ thống: {e}")

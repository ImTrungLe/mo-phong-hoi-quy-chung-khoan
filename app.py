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
# TAB 4: THỰC NGHIỆM DỮ LIỆU THỰC TẾ
# ==========================================
with tab4:
    st.header("⚡ Thực Nghiệm Thuật Toán Định Lượng Với Dữ Liệu Thật")
    st.write("Nhập mã cổ phiếu để hệ thống tự động tải dữ liệu lịch sử và dùng phép toán Đại số tuyến tính giải ngược ra các tham số của tiến trình Ornstein-Uhlenbeck.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Nhập mã cổ phiếu (Yahoo Finance):", "AAPL")
        st.caption("Ví dụ: AAPL (Apple), MSFT (Microsoft), VCB.VN (Vietcombank), FPT.VN (FPT)")
        
        days_lookback = st.slider("Thời gian phân tích (ngày quá khứ)", 100, 500, 252)
        run_btn = st.button("Chạy tính toán tham số")
        
    with col2:
        if run_btn or ticker:
            try:
                # 1. Tải dữ liệu từ Yahoo Finance
                end_date = datetime.today()
                start_date = end_date - timedelta(days=days_lookback)
                
                with st.spinner('Đang tải dữ liệu từ Yahoo Finance...'):
                    data = yf.download(ticker, start=start_date, end=end_date)
                
                if data.empty:
                    st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này. Hãy kiểm tra lại ký hiệu mã.")
                else:
                    # Lấy chuỗi giá đóng cửa
                    prices = data['Close'].values.flatten()
                    dates = data.index
                    
                    # 2. Xây dựng toán học ước lượng tham số (Model Calibration via OLS)
                    X_prev = prices[:-1]
                    X_curr = prices[1:]
                    
                    # Áp dụng Đại số tuyến tính tìm ma trận hệ số bằng OLS [X_prev, 1]
                    A = np.vstack([X_prev, np.ones(len(X_prev))]).T
                    m, c = np.linalg.lstsq(A, X_curr, rcond=None)[0]
                    
                    # Tính toán phần dư (Residuals) để tìm phương sai sai số
                    residuals = X_curr - (m * X_prev + c)
                    res_variance = np.var(residuals, ddof=2)
                    
                    dt_step = 1.0  # Bước thời gian rời rạc là 1 ngày
                    
                    if m > 0 and m < 1: # Đảm bảo tính dừng
                        calculated_theta = -np.log(m) / dt_step
                        calculated_mu = c / (1 - m)
                        calculated_sigma = np.sqrt(res_variance * 2 * calculated_theta / (1 - m**2))
                        
                        # 3. Trực quan hóa kết quả lên đồ thị
                        fig4 = go.Figure()
                        fig4.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name=f'🔵 Giá thực tế {ticker}', line=dict(color='#1f77b4')))
                        
                        # Trục trung bình
                        fig4.add_shape(type="line", x0=dates[0], y0=calculated_mu, x1=dates[-1], y1=calculated_mu, line=dict(color="Red", width=2, dash="dash"))
                        fig4.add_trace(go.Scatter(x=[dates[-1]], y=[calculated_mu], mode='text', text=['🔴 Trục cân bằng'], textposition='top left', showlegend=False))
                        
                        # Biên trên / Biên dưới
                        upper_band = calculated_mu + 2 * (calculated_sigma / np.sqrt(2 * calculated_theta))
                        lower_band = calculated_mu - 2 * (calculated_sigma / np.sqrt(2 * calculated_theta))
                        
                        fig4.add_shape(type="line", x0=dates[0], y0=upper_band, x1=dates[-1], y1=upper_band, line=dict(color="rgba(255,0,0,0.5)", width=2, dash="dot"))
                        fig4.add_trace(go.Scatter(x=[dates[-1]], y=[upper_band], mode='text', text=['Biên Quá Mua (+2σ)'], textposition='top left', showlegend=False, textfont=dict(color="red")))
                        
                        fig4.add_shape(type="line", x0=dates[0], y0=lower_band, x1=dates[-1], y1=lower_band, line=dict(color="rgba(0,128,0,0.5)", width=2, dash="dot"))
                        fig4.add_trace(go.Scatter(x=[dates[-1]], y=[lower_band], mode='text', text=['Biên Quá Bán (-2σ)'], textposition='bottom left', showlegend=False, textfont=dict(color="green")))
                        
                        fig4.update_layout(title=f"Định lượng vùng giao dịch của {ticker}", xaxis_title="Ngày giao dịch", yaxis_title="Giá", height=500, showlegend=True)
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        # Hiển thị các khối hộp thông số
                        st.subheader("📋 Kết quả giải mã hệ thống SDE từ dữ liệu thực:")
                        cm1, cm2, cm3 = st.columns(3)
                        cm1.metric(label="🎯 Trục cân bằng dài hạn (μ)", value=f"{calculated_mu:.2f}")
                        cm2.metric(label="⚡ Tốc độ hồi quy (θ)", value=f"{calculated_theta:.4f}")
                        cm3.metric(label="🎲 Độ lệch chuẩn rủi ro", value=f"{calculated_sigma:.4f}")
                        
                        st.info(f"💡 **Phân tích:** Hệ thống đã tự động lấy ma trận đặc trưng của chuỗi giá {ticker}. Bằng cách giải bài toán bình phương tối thiểu, ta tìm được trục cân bằng lý thuyết ở mức {calculated_mu:.2f}. Các đường đứt nét mô phỏng giới hạn sai số cho phép định vị các vùng Quá mua / Quá bán.")
                    else:
                        st.warning("⚠️ Cổ phiếu này đang trong một xu hướng tăng trưởng hoặc suy thoái quá mạnh (hệ số tự hồi quy >= 1). Hệ thống không thể tìm thấy điểm hội tụ trung bình hồi quy tĩnh.")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi hệ thống khi phân tích dữ liệu: {e}")

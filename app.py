import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Cấu hình trang
st.set_page_config(page_title="Mô phỏng Hồi quy Chứng khoán", layout="wide")
st.title("📊 Ứng Dụng Mô Phỏng Các Mô Hình Hồi Quy Trong Tài Chính")

# Chia các tab nội dung theo đúng cấu trúc nghiên cứu
tab1, tab2, tab3 = st.tabs([
    "1. Hồi quy tuyến tính tĩnh", 
    "2. Mô hình tự hồi quy AR(1)", 
    "3. Tiến trình ngẫu nhiên liên tục O-U"
])

# ==========================================
# TAB 1: HỒI QUY TUYẾN TÍNH TĨNH
# ==========================================
with tab1:
    st.header("Hồi quy tuyến tính tĩnh (Linear Regression)")
    st.latex(r"Y = a + bX") # [cite: 79]
    st.write("Sử dụng phương pháp Bình phương tối thiểu (OLS) để vẽ một đường thẳng xu hướng xuyên qua các điểm giá[cite: 78].")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        n_points = st.slider("Số phiên giao dịch (N)", 50, 500, 200, step=50)
        slope = st.slider("Hệ số góc b (Độ dốc xu hướng)", -0.5, 0.5, 0.1, step=0.05)
        intercept = st.slider("Hệ số chặn a", 10, 100, 50)
        noise_level = st.slider("Độ nhiễu thị trường", 1, 20, 5)
        
    with col2:
        # Tạo dữ liệu giả lập xu hướng
        X = np.arange(n_points)
        pure_Y = intercept + slope * X
        noise = np.random.normal(0, noise_level, n_points)
        Y = pure_Y + noise
        
        # Vẽ đồ thị bằng Plotly
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Giá thực tế (Có nhiễu)'))
        fig1.add_trace(go.Scatter(x=X, y=pure_Y, mode='lines', name='Đường hồi quy lý thuyết', line=dict(color='red', width=2)))
        fig1.update_layout(title="Mô phỏng đường xu hướng tĩnh", xaxis_title="Thời gian (Phiên)", yaxis_title="Giá cổ phiếu", height=450)
        st.plotly_chart(fig1, use_container_width=True)

# ==========================================
# TAB 2: MÔ HÌNH TỰ HỒI QUY RỜI RẠC AR(1)
# ==========================================
with tab2:
    st.header("Mô hình tự hồi quy rời rạc AR(1)")
    st.latex(r"X_t = \alpha + \beta X_{t-1} + \epsilon_t") # [cite: 85]
    st.write("Mô hình thừa nhận tính phụ thuộc tuyến tính mang bản chất Markovian: giá trị tài sản ở hiện tại chịu tác động trực tiếp từ giá trị liền kề trước đó của chính nó[cite: 84].")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        alpha = st.slider("Hệ số tự do alpha", 1, 10, 5)
        beta = st.slider("Hệ số tự hồi quy beta", -1.5, 1.5, 0.7, step=0.05)
        st.write("💡 *Điều kiện tồn tại trung bình hồi quy: |beta| < 1. Nếu |beta| >= 1, chuỗi sẽ rơi vào trạng thái bước ngẫu nhiên hoặc bùng nổ[cite: 86].*")
        
    with col2:
        # Sinh chuỗi dữ liệu AR(1)
        np.random.seed(42)
        ar_data = np.zeros(200)
        ar_data[0] = 50 # Giá khởi điểm
        for t in range(1, 200):
            ar_data[t] = alpha + beta * ar_data[t-1] + np.random.normal(0, 2)
            # Tránh lỗi bùng nổ quá lớn làm hỏng đồ thị khi vẽ
            if abs(ar_data[t]) > 10000:
                ar_data[t:] = np.nan
                break
                
        # Tính toán đường trung bình dài hạn nếu thỏa mãn điều kiện dừng
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=ar_data, mode='lines', name='Chuỗi giá AR(1)'))
        if abs(beta) < 1:
            mean_long_term = alpha / (1 - beta)
            fig2.add_shape(type="line", x0=0, y0=mean_long_term, x1=200, y1=mean_long_term, line=dict(color="Green", dash="dash"), name="Mức cân bằng")
            st.success(f"Chuỗi đạt trạng thái dừng. Trục trung bình hồi quy dài hạn: {mean_long_term:.2f}")
        else:
            st.error("Chuỗi không dừng (Bùng nổ hoặc Bước ngẫu nhiên)! Không tồn tại xu hướng hội tụ về mức cân bằng[cite: 86].")
            
        fig2.update_layout(title="Mô phỏng tính dừng và hội tụ của mô hình rời rạc", xaxis_title="Thời gian (t)", yaxis_title="Giá trị Xt", height=450)
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 3: TIẾN TRÌNH NGẪU NHIÊN LIÊN TỤC O-U
# ==========================================
with tab3:
    st.header("Tiến trình ngẫu nhiên liên tục Ornstein - Uhlenbeck (O-U Process)")
    st.latex(r"dX_t = \theta(\mu - X_t)dt + \sigma dW_t") # [cite: 90]
    st.write("Mô hình hóa dòng chảy liên tục của thị trường, mô tả chính xác lực kéo hồi quy về mức cân bằng dài hạn lý thuyết[cite: 94, 100].")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        theta = st.slider("Hệ số tốc độ hồi quy (theta) [cite: 93]", 0.05, 1.00, 0.30, step=0.05)
        mu = st.slider("Mức trung bình dài hạn (mu) [cite: 94]", 50, 150, 100)
        sigma = st.slider("Hệ số biến động / Độ rủi ro (sigma) [cite: 95]", 1, 20, 5)
        
    with col2:
        # Giải phương trình vi phân ngẫu nhiên bằng phương pháp Euler-Maruyama
        T = 10.0
        N = 500
        dt = T / N
        t_space = np.linspace(0, T, N)
        
        X_ou = np.zeros(N)
        X_ou[0] = 60 # Giá ban đầu lệch xa khỏi mu để thấy rõ lực kéo
        
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            X_ou[i] = X_ou[i-1] + theta * (mu - X_ou[i-1]) * dt + sigma * dW # [cite: 90]
            
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t_space, y=X_ou, mode='lines', name='Quỹ đạo giá liên tục'))
        fig3.add_shape(type="line", x0=0, y0=mu, x1=T, y1=mu, line=dict(color="orange", width=2, dash="dash"), name="Mức trung bình mu")
        fig3.update_layout(title="Mô phỏng lực kéo liên tục của tiến trình O-U", xaxis_title="Thời gian liên tục (t)", yaxis_title="Giá tài sản Xt", height=450)
        st.plotly_chart(fig3, use_container_width=True)
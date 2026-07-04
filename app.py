"""
Math Model 3D - Volleyball Flight Simulation
Главный файл приложения Streamlit. 
Отвечает за отрисовку интерфейса, синхронизацию состояния и интеграцию с физическим движком.
"""
import streamlit as st
import numpy as np
import base64
from src.input_handler import SimulationParams, ServeType
from src.physics import solve_trajectory_3d, calculate_impact_force
from src.visualization import plot_speed_2d
from src.component import volleyball_3d
from src.analyzer import evaluate_serve
from src.i18n import TEXT
from src.config import BALL_MODELS

# Select language via session state or default to Russian
if "lang" not in st.session_state:
    st.session_state.lang = "ru"
t = TEXT[st.session_state.lang]

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set page config
st.set_page_config(page_title=t["page_title"], page_icon="assets/BeeLogo.ico", layout="wide")

# Custom CSS for Constructivism style with Auto Dark/Light theme support
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Oswald', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Remove all border radius and add harsh borders */
    .stButton>button, .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 0px !important;
        border: 2px solid var(--text-color) !important;
        box-shadow: 4px 4px 0px var(--text-color) !important;
        transition: all 0.1s;
    }
    
    .stButton>button:active {
        box-shadow: 0px 0px 0px var(--text-color) !important;
        transform: translate(4px, 4px);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: var(--background-color);
        border: 4px solid var(--text-color);
        padding: 15px;
        box-shadow: 6px 6px 0px #E3000F;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 700;
        color: var(--text-color);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #E3000F;
    }
    
    /* Alerts for Analytics */
    [data-testid="stAlert"] {
        border-radius: 0px !important;
        border: 4px solid var(--text-color) !important;
        box-shadow: 6px 6px 0px #E3000F !important;
        color: var(--text-color) !important;
        background-color: var(--background-color) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Oswald', sans-serif !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
        text-transform: uppercase;
    }
    
    h1 {
        border-bottom: 8px solid #E3000F;
        display: inline-block;
        padding-bottom: 5px;
        margin-bottom: 30px;
    }
    
    /* Sidebar removal */
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Hide ONLY the Deploy button, keep the settings menu */
    .stDeployButton {display: none;}
    footer {visibility: hidden;}

    hr {
        border-top: 4px solid var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# CSS для кнопки языка (узкий селектор, чтобы не ломать другие колонки)
st.markdown("""
<style>
    /* Выравнивание кнопки языка вправо (только в header-контейнере) */
    [data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:last-child {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Header Row
header_col1, header_col2 = st.columns([6, 1])

with header_col1:
    try:
        logo_base64 = get_base64_of_bin_file("assets/BeeLogo.svg")
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 0.5rem; margin-top: -1rem;">
            <img src="data:image/svg+xml;base64,{logo_base64}" width="65" height="65" style="object-fit: contain;" />
            <h1 style="margin: 0; padding: 0; font-family: 'Oswald', sans-serif !important; font-size: 3.5rem; text-transform: uppercase;">{t['title']}</h1>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.title(t["title"])

with header_col2:
    def toggle_lang():
        st.session_state.lang = "en" if st.session_state.lang == "ru" else "ru"
    
    current_lang_str = "RU" if st.session_state.lang == "ru" else "EN"
    st.button(f"🌐 {current_lang_str}", on_click=toggle_lang, key="lang_btn")

st.markdown(t["subtitle"])

# ----------------- SESSION STATE SYNC ----------------- #
def init_state(key, default):
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = default
    if f"{key}_num" not in st.session_state:
        st.session_state[f"{key}_num"] = default

def sync_slider(key):
    st.session_state[f"{key}_num"] = st.session_state[f"{key}_slider"]

def sync_num(key):
    st.session_state[f"{key}_slider"] = st.session_state[f"{key}_num"]

# Init defaults
init_state("v0", 90.0) # 90 km/h is a good default (25 m/s)
init_state("alpha", 10.0)
init_state("azimuth", 0.0)
init_state("y0", 2.5)
init_state("start_z", 0.0)
init_state("start_z", 0.0)
init_state("spin", 800.0)
init_state("spin_angle", 0.0)
if "ball_type" not in st.session_state:
    st.session_state.ball_type = "MIKASA_V200W"
init_state("mass", BALL_MODELS["MIKASA_V200W"].mass)
init_state("cd", BALL_MODELS["MIKASA_V200W"].cd)
# ------------------------------------------------------ #

# Layout: 2 columns for Main (controls) and Visualization
main_col, viz_col = st.columns([1, 2], gap="large")

with main_col:
    st.markdown(t["params"])
    
    serve_type_str = st.selectbox(t["serve_type"], [t["serve_topspin"], t["serve_float"]])
    serve_type = ServeType.TOPSPIN if t["serve_topspin"] == serve_type_str else ServeType.FLOAT
    
    st.markdown("Модель мяча") # Add translations later if needed
    ball_options = list(BALL_MODELS.keys())
    ball_type = st.selectbox("Мяч", ball_options, index=ball_options.index(st.session_state.ball_type), label_visibility="collapsed")
    
    if ball_type != st.session_state.ball_type:
        st.session_state.ball_type = ball_type
        if ball_type != "CUSTOM":
            st.session_state.mass_num = BALL_MODELS[ball_type].mass
            st.session_state.mass_slider = BALL_MODELS[ball_type].mass
            st.session_state.cd_num = BALL_MODELS[ball_type].cd
            st.session_state.cd_slider = BALL_MODELS[ball_type].cd
        st.rerun()
    
    st.markdown("---")
    
    # Скорость
    st.markdown(t["v0"])
    c1, c2 = st.columns([3, 1])
    c1.slider("v0_s", 30.0, 150.0, key="v0_slider", on_change=sync_slider, args=("v0",), label_visibility="collapsed")
    c2.number_input("v0_n", 30.0, 150.0, key="v0_num", on_change=sync_num, args=("v0",), label_visibility="collapsed")
    
    # Угол вылета (вертикальный)
    st.markdown(t["alpha"])
    c1, c2 = st.columns([3, 1])
    c1.slider("alpha_s", -15.0, 45.0, key="alpha_slider", on_change=sync_slider, args=("alpha",), label_visibility="collapsed")
    c2.number_input("alpha_n", -15.0, 45.0, key="alpha_num", on_change=sync_num, args=("alpha",), label_visibility="collapsed")
    
    # Направление подачи (горизонтальный азимут)
    st.markdown(t["azimuth"])
    c1, c2 = st.columns([3, 1])
    c1.slider("azimuth_s", -30.0, 30.0, key="azimuth_slider", on_change=sync_slider, args=("azimuth",), label_visibility="collapsed")
    c2.number_input("azimuth_n", -30.0, 30.0, key="azimuth_num", on_change=sync_num, args=("azimuth",), label_visibility="collapsed")
    
    # Высота
    st.markdown(t["y0"])
    c1, c2 = st.columns([3, 1])
    c1.slider("y0_s", 1.0, 4.0, key="y0_slider", on_change=sync_slider, args=("y0",), label_visibility="collapsed")
    c2.number_input("y0_n", 1.0, 4.0, key="y0_num", on_change=sync_num, args=("y0",), label_visibility="collapsed")
    
    # Позиция подающего
    st.markdown(t["start_z"])
    c1, c2 = st.columns([3, 1])
    c1.slider("start_z_s", -4.5, 4.5, key="start_z_slider", on_change=sync_slider, args=("start_z",), label_visibility="collapsed")
    c2.number_input("start_z_n", -4.5, 4.5, key="start_z_num", on_change=sync_num, args=("start_z",), label_visibility="collapsed")
    
    if serve_type == ServeType.FLOAT:
        st.info(t["float_tip"])
        
    st.markdown(t["spin"])
    c1, c2 = st.columns([3, 1])
    c1.slider("spin_s", 0.0, 1500.0, key="spin_slider", on_change=sync_slider, args=("spin",), label_visibility="collapsed")
    c2.number_input("spin_n", 0.0, 1500.0, key="spin_num", on_change=sync_num, args=("spin",), label_visibility="collapsed")
    current_spin = st.session_state.spin_num
    
    st.markdown(t["spin_angle"])
    c1, c2 = st.columns([3, 1])
    c1.slider("spin_angle_s", -45.0, 45.0, key="spin_angle_slider", on_change=sync_slider, args=("spin_angle",), label_visibility="collapsed")
    c2.number_input("spin_angle_n", -45.0, 45.0, key="spin_angle_num", on_change=sync_num, args=("spin_angle",), label_visibility="collapsed")
    current_spin_angle = st.session_state.spin_angle_num
        
    with st.expander(t["additional"]):
        st.markdown(t["mass"])
        c1, c2 = st.columns([3, 1])
        c1.slider("mass_s", 0.20, 0.35, key="mass_slider", on_change=sync_slider, args=("mass",), label_visibility="collapsed")
        c2.number_input("mass_n", 0.20, 0.35, key="mass_num", on_change=sync_num, args=("mass",), label_visibility="collapsed")
        
        st.markdown(t["cd"])
        c1, c2 = st.columns([3, 1])
        c1.slider("cd_s", 0.0, 1.0, key="cd_slider", on_change=sync_slider, args=("cd",), label_visibility="collapsed")
        c2.number_input("cd_n", 0.0, 1.0, key="cd_num", on_change=sync_num, args=("cd",), label_visibility="collapsed")

# Build params
params = SimulationParams(
    ball_type=st.session_state.ball_type,
    mass=st.session_state.mass_num,
    v0=st.session_state.v0_num / 3.6,
    alpha_deg=st.session_state.alpha_num,
    azimuth_deg=st.session_state.azimuth_num,
    y0=st.session_state.y0_num,
    start_z=st.session_state.start_z_num,
    serve_type=serve_type,
    spin_rpm=current_spin,
    spin_angle_deg=current_spin_angle,
    cd=st.session_state.cd_num
)

try:
    time_arr, x, y, z, vx, vy, vz = solve_trajectory_3d(params)
    
    # Calculate speed array and find index of max speed
    speed_ms = (vx**2 + vy**2 + vz**2)**0.5
    speed_kmh = speed_ms * 3.6
    idx_max_v = int(np.argmax(speed_kmh))
    
    impact_force = calculate_impact_force(st.session_state.v0_num / 3.6, st.session_state.mass_num)
    # Find actual landing point (first time it hits the ground)
    landing_idx = __import__('numpy').where(y <= 0.15)[0]
    distance = x[landing_idx[0]] if len(landing_idx) > 0 else x[-1]
    flight_time = time_arr[-1]
    max_height = max(y)
    
    with viz_col:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t["dist"], f"{distance:.2f} {t['unit_m']}")
        m2.metric(t["height"], f"{max_height:.2f} {t['unit_m']}")
        m3.metric(t["time"], f"{flight_time:.2f} {t['unit_s']}")
        m4.metric(t["force"], f"{impact_force:.1f} {t['unit_n']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        # Using the new React-based 3D Component!
        volleyball_3d(time_arr, x, y, z, idx_max_v)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # Отрезаем данные после первого касания пола, чтобы график скорости не показывал резкий скачок отскока
        end_idx = landing_idx[0] + 1 if len(landing_idx) > 0 else len(time_arr)
        fig_speed = plot_speed_2d(time_arr[:end_idx], speed_kmh[:end_idx], t)
        st.plotly_chart(fig_speed, use_container_width=True, theme="streamlit")
        
        import pandas as pd
        df = pd.DataFrame({"Time(s)": time_arr, "X(m)": x, "Y_Height(m)": y, "Z_Width(m)": z, "Speed(km/h)": speed_kmh})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Скачать CSV (Траектория)", csv, "trajectory.csv", "text/csv")
        
        # Аналитика
        st.markdown(f"### {t['an_title']}")
        analysis_results = evaluate_serve(x, y, z, time_arr, speed_kmh, serve_type, t)
        for res in analysis_results:
            if res["status"] == "success":
                st.success(f"**{res['param']}**: {res['comment']}")
            elif res["status"] == "error":
                st.error(f"**{res['param']}**: {res['comment']}")
            else:
                st.warning(f"**{res['param']}**: {res['comment']}")
                
        # Продвинутая аналитика (Этап 2)
        st.markdown("### Цифровой тренер (Продвинутая аналитика)")
        tab_mc, tab_opt = st.tabs(["Монте-Карло (Стабильность)", "Поиск оптимальной подачи"])
        
        with tab_mc:
            st.write("Симуляция 30 подач со случайными отклонениями (±5% скорости, ±2° угла). Позволяет оценить процент успешного попадания при неидеальном исполнении.")
            if st.button("Запустить симуляцию Монте-Карло"):
                from src.analyzer import run_monte_carlo
                from src.visualization import plot_monte_carlo
                with st.spinner("Симуляция..."):
                    success_rate, points = run_monte_carlo(params, n=30)
                    st.metric("Вероятность попадания (Success Rate)", f"{success_rate:.1f}%")
                    fig_mc = plot_monte_carlo(points)
                    st.plotly_chart(fig_mc, use_container_width=True, theme="streamlit")
                    
        with tab_opt:
            st.write("Укажите желаемую точку падения мяча на стороне противника, и алгоритм подберет идеальную скорость и угол.")
            opt_c1, opt_c2 = st.columns(2)
            target_x = opt_c1.number_input("Целевой X (длина, 9-18м)", min_value=9.0, max_value=18.0, value=17.0, step=0.5)
            target_z = opt_c2.number_input("Целевой Z (ширина, от -4.5 до 4.5м)", min_value=-4.5, max_value=4.5, value=4.0, step=0.5)
            
            if st.button("Найти параметры"):
                from src.analyzer import optimize_serve
                with st.spinner("Алгоритм Nelder-Mead ищет решение..."):
                    opt_res = optimize_serve(params, target_x, target_z)
                    if opt_res["success"]:
                        st.success("Параметры успешно найдены!")
                        st.write(f"**Скорость:** {opt_res['v0']*3.6:.1f} км/ч")
                        st.write(f"**Вертикальный угол:** {opt_res['alpha_deg']:.1f}°")
                        st.write(f"**Горизонтальный азимут:** {opt_res['azimuth_deg']:.1f}°")
                        
                        # Apply button
                        if st.button("Применить эти параметры"):
                            st.session_state.v0_num = opt_res['v0'] * 3.6
                            st.session_state.v0_slider = opt_res['v0'] * 3.6
                            st.session_state.alpha_num = opt_res['alpha_deg']
                            st.session_state.alpha_slider = opt_res['alpha_deg']
                            st.session_state.azimuth_num = opt_res['azimuth_deg']
                            st.session_state.azimuth_slider = opt_res['azimuth_deg']
                            st.rerun()
                    else:
                        st.error("Не удалось найти стабильное решение для заданной точки с текущими параметрами высоты и вращения. Попробуйте изменить точку или базовые настройки.")

except Exception as e:
    st.error(f"{t['error']} {e}")


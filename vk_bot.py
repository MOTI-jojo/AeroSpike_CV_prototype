import os
import re
import numpy as np
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.upload import VkUpload
import io

# Импортируем нашу физику
from src.input_handler import SimulationParams, ServeType
from src.physics import solve_trajectory_3d
from src.config import BALL_MODELS
from src.visualization import plot_speed_2d, plot_trajectory_3d
from src.i18n import TEXT

# Загружаем переменные окружения, если используется dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

VK_TOKEN = os.getenv("VK_BOT_TOKEN")

if not VK_TOKEN:
    print("ОШИБКА: Не задан токен VK_BOT_TOKEN. Установите переменную окружения или создайте файл .env")
    exit(1)

def get_text():
    # Используем русский по умолчанию для бота
    return TEXT["ru"]

def process_serve(v0_kmh: float, alpha_deg: float, azimuth_deg: float):
    # Подготавливаем параметры
    params = SimulationParams(
        ball_type="MIKASA_V200W",
        mass=BALL_MODELS["MIKASA_V200W"].mass,
        cd=BALL_MODELS["MIKASA_V200W"].cd,
        v0=v0_kmh / 3.6,
        alpha_deg=alpha_deg,
        azimuth_deg=azimuth_deg,
        y0=2.5,  # Стандартная высота
        start_z=0.0,
        serve_type=ServeType.TOPSPIN,
        spin_rpm=800.0,
        spin_angle_deg=0.0,
        wind_speed=0.0,
        wind_direction_deg=0.0
    )
    
    t_dict = get_text()
    
    try:
        time_arr, x, y, z, vx, vy, vz = solve_trajectory_3d(params)
        
        speed_ms = (vx**2 + vy**2 + vz**2)**0.5
        speed_kmh = speed_ms * 3.6
        idx_max_v = int(np.argmax(speed_kmh))
        
        # Находим приземление
        landing_idx = np.where(y <= 0.15)[0]
        if len(landing_idx) > 0:
            idx_land = landing_idx[0]
            distance = x[idx_land]
            flight_time = time_arr[idx_land]
            end_idx = idx_land + 1
        else:
            distance = x[-1]
            flight_time = time_arr[-1]
            end_idx = len(time_arr)
            
        max_height = max(y)
        
        # Определение статуса
        status = "В площадке (Эйс!) ✅"
        if distance > 18.0 or abs(z[end_idx-1]) > 4.5:
            status = "Аут ❌"
        
        # Проверка сетки
        idx_net = np.where(x >= 9.0)[0]
        if len(idx_net) > 0 and y[idx_net[0]] <= 2.55:
            status = "Сетка ❌"
            
        msg_text = (
            f"🏐 Симуляция завершена!\n\n"
            f"Ввод: Скорость {v0_kmh} км/ч, Угол {alpha_deg}°, Азимут {azimuth_deg}°\n\n"
            f"📊 Результаты:\n"
            f"Дальность: {distance:.2f} м\n"
            f"Макс. высота: {max_height:.2f} м\n"
            f"Время полёта: {flight_time:.2f} с\n"
            f"Итог: {status}"
        )
        
        # Генерируем красивый 2D график (график скорости)
        fig = plot_speed_2d(time_arr[:end_idx], speed_kmh[:end_idx], t_dict)
        
        # Сохраняем в буфер памяти (не засоряем диск)
        img_bytes = fig.to_image(format="png", engine="kaleido", width=800, height=500)
        img_io = io.BytesIO(img_bytes)
        img_io.name = "plot.png"
        
        return msg_text, img_io
        
    except Exception as e:
        return f"Произошла ошибка при симуляции: {str(e)}", None

def main():
    print("Запуск бота VK...")
    vk_session = vk_api.VkApi(token=VK_TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)
    upload = VkUpload(vk_session)
    
    print("Бот успешно запущен и слушает сообщения!")
    
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            text = event.text.lower()
            user_id = event.user_id
            
            # Обработка команды "подача"
            if text.startswith("подача"):
                # Ищем числа в сообщении
                nums = re.findall(r'-?\d+(?:\.\d+)?', text)
                if len(nums) >= 3:
                    v0 = float(nums[0])
                    alpha = float(nums[1])
                    azimuth = float(nums[2])
                    
                    vk.messages.send(
                        user_id=user_id,
                        message=f"⏳ Считаю физику (v0={v0}, alpha={alpha}, az={azimuth})...",
                        random_id=0
                    )
                    
                    msg_resp, img_io = process_serve(v0, alpha, azimuth)
                    
                    attachment = None
                    if img_io:
                        # Загружаем фото на сервер VK
                        photo = upload.photo_messages(photos=img_io)[0]
                        attachment = f"photo{photo['owner_id']}_{photo['id']}"
                    
                    vk.messages.send(
                        user_id=user_id,
                        message=msg_resp,
                        attachment=attachment,
                        random_id=0
                    )
                    
                else:
                    vk.messages.send(
                        user_id=user_id,
                        message="ОШИБКА: Неверный формат.\nИспользование: Подача <скорость> <угол> <азимут>\nПример: Подача 110 15 0",
                        random_id=0
                    )
            
            elif text in ["привет", "начать", "help", "помощь"]:
                help_text = (
                    "👋 Привет! Я бот волейбольного симулятора AeroSpike.\n\n"
                    "Чтобы смоделировать полёт мяча, напиши команду в формате:\n"
                    "Подача <скорость> <угол> <азимут>\n\n"
                    "Примеры:\n"
                    "👉 Подача 110 12 0\n"
                    "👉 Подача 65 15 5\n\n"
                    "Скорость в км/ч, углы в градусах."
                )
                vk.messages.send(
                    user_id=user_id,
                    message=help_text,
                    random_id=0
                )

if __name__ == "__main__":
    main()

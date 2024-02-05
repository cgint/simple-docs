from datetime import datetime

def cur_simple_date_time_sec() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

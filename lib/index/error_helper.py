from lib import constants
from lib.index.helper import cur_simple_date_time_sec

def write_error_to_file(e: Exception, msg: str = None):
    from traceback import format_exception
    with open(constants.error_file, "a") as f:
        f.write("========================================================================\n")
        f.write(f"================= {cur_simple_date_time_sec()} ==================\n")
        f.write("========================================================================\n")
        f.write(msg+"\n") if msg else None
        f.write("Trace: "+"".join(format_exception(e))+"\n")
        f.write("========================================================================\n")

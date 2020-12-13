import logging
import sys
from mod1 import mod1fn


def set_logger(filepath):
    root = logging.basicConfig(
        filename=str(filepath),
        filemode="w",  # will rewrite on each run
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    return root


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    filename=str("testlog2.log"),
    filemode="w",  # will rewrite on each run
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# logging.setFormatter(formatter)
logger.info("adsf")
print("from main..")

mod1fn()

# root = logging.getLogger()
# root.setLevel(logging.DEBUG)


# root.info("asdf")

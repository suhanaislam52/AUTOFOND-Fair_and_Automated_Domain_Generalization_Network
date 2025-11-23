from src.datasets.pacs import PACS
from src.datasets.vlcs import VLCS
from src.datasets.officehome import OfficeHome
from src.datasets.wilds import WILDSCamelyon

DATASETS = {
    "PACS": PACS,
    "VLCS": VLCS,
    "OfficeHome": OfficeHome,
    "WILDSCamelyon": WILDSCamelyon
}

from datetime import timedelta
from pathlib import Path
import requests

from ..auth.handler import SecretStore

def generate_urls_from_pattern(start_dt, end_dt, pattern, interval:timedelta):
    """
    Generate URLs by filling a datetime pattern at regular intervals.
    
    Args:
        start_dt (datetime): Start datetime (inclusive)
        end_dt (datetime): End datetime (inclusive)
        pattern (str): URL pattern with placeholders (e.g., {year}, {month}, {datetime}, etc.)
        interval_minutes (int): Time step in minutes
    
    Returns:
        List[str]: List of formatted URLs
    """
    urls = []

    current = start_dt
    while current <= end_dt:
        url = pattern.format(
            year=current.year,
            month=f"{current.month:02d}",
            day=f"{current.day:02d}",
            hour=f"{current.hour:02d}",
            minute=f"{current.minute:02d}",
            datetime=current.strftime("%Y%m%d%H%M"),
        )
        urls.append(url)
        current += interval

    return urls


class LSASAFCrawler():

    def __init__(self, pattern:str|None=None):
        self.auth = None
        if SecretStore().exists():
            self.auth = SecretStore().get_key('lsasaf')
        self.pattern = pattern if pattern else (
            "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/"
            "{year}/{month}/{day}/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_{datetime}.nc"
        )

    def url_from_pattern(self, start_time, end_time):
        return generate_urls_from_pattern(start_time, end_time, self.pattern, timedelta(days=1))
    
    def load_granules(self, urls:list[str], out_dir:str | None = None, auth:tuple[str]|None = None):
        for url in urls:
            self.load_granule(url, out_dir, auth)

    def load_granule(self, url:str, out_dir:str | None = None, auth:tuple[str]|None = None):
        if not auth and not self.auth:
            raise ValueError('provide auth for datalsasaf')
        auth = auth if auth else self.auth
        out_dir=out_dir if out_dir else ''
        basename = Path(url).name
        
        out_path = Path(out_dir) / basename
        if out_path.exists():
            print(f'File {basename} exists at {out_dir}!')
            return
        
        response = requests.get(url, auth=auth, timeout=60)
        
        if response.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(response.content)
        else:
            print("Failed:", response.status_code)
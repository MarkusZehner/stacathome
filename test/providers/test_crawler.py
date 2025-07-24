import pytest
from datetime import datetime

from stacathome.providers.crawler import LSASAFCrawler


class TestCrawler:

    @pytest.mark.remote
    def test_create_url_by_pattern(self):
        start = datetime(2023, 12, 1, 0, 0)
        end = datetime(2023, 12, 10, 0, 0)

        EXPECTED_URLS = {
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/01/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312010000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/02/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312020000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/03/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312030000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/04/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312040000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/05/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312050000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/06/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312060000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/07/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312070000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/08/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312080000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/09/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312090000.nc',
            'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MEM/NETCDF/2023/12/10/NETCDF4_LSASAF_MSG_EMMAPS_MSG-Disk_202312100000.nc',
        }

        urls = LSASAFCrawler().url_from_pattern(start, end)

        assert set(urls) == EXPECTED_URLS

        #LSASAFCrawler().load_granules(urls[:2], out_dir='/Net/Groups/BGI/work_5/scratch/mzehner/temp_test')
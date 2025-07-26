import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import tempfile
import sys
from io import StringIO

from select_stock import (
    load_data, load_config, load_appendix_codes, load_hs300_codes,
    load_a500_codes, load_combined_codes, instantiate_selector, main
)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return """date,open,high,low,close,volume
2023-01-01,10.0,11.0,9.5,10.5,1000000
2023-01-02,10.5,11.5,10.0,11.0,1200000
2023-01-03,11.0,12.0,10.5,11.5,1100000"""


@pytest.fixture
def temp_data_dir(sample_csv_data):
    """Create temporary directory with sample CSV files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        # Create sample CSV files
        (data_dir / "000001.csv").write_text(sample_csv_data)
        (data_dir / "000002.csv").write_text(sample_csv_data)
        (data_dir / "600000.csv").write_text(sample_csv_data)
        
        yield data_dir


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "selectors": [{
            "class": "BBIKDJSelector",
            "alias": "测试选择器",
            "activate": True,
            "params": {
                "j_threshold": 1,
                "bbi_q_threshold": 0.1,
                "j_q_threshold": 0.1
            }
        }]
    }


@pytest.fixture
def sample_appendix():
    """Sample appendix.json data"""
    return {
        "data": ["000001", "000002", "600000", "600036"]
    }


class TestLoadData:
    def test_load_data_success(self, temp_data_dir):
        """Test successful data loading"""
        codes = ["000001", "000002"]
        result = load_data(temp_data_dir, codes)
        
        assert len(result) == 2
        assert "000001" in result
        assert "000002" in result
        assert isinstance(result["000001"], pd.DataFrame)
        assert len(result["000001"]) == 3
        assert "date" in result["000001"].columns

    def test_load_data_missing_file(self, temp_data_dir, caplog):
        """Test loading data with missing files"""
        codes = ["000001", "999999"]  # 999999 doesn't exist
        result = load_data(temp_data_dir, codes)
        
        assert len(result) == 1
        assert "000001" in result
        assert "999999" not in result
        assert "999999.csv 不存在，跳过" in caplog.text

    def test_load_data_empty_codes(self, temp_data_dir):
        """Test loading data with empty codes list"""
        result = load_data(temp_data_dir, [])
        assert len(result) == 0


class TestLoadConfig:
    def test_load_config_with_selectors_key(self, sample_config):
        """Test loading config with 'selectors' key"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config, f)
            config_path = Path(f.name)
        
        try:
            result = load_config(config_path)
            assert len(result) == 1
            assert result[0]["class"] == "BBIKDJSelector"
            assert result[0]["alias"] == "测试选择器"
        finally:
            config_path.unlink()

    def test_load_config_list_format(self):
        """Test loading config as direct list"""
        config_list = [{
            "class": "TestSelector",
            "params": {"test": True}
        }]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_list, f)
            config_path = Path(f.name)
        
        try:
            result = load_config(config_path)
            assert len(result) == 1
            assert result[0]["class"] == "TestSelector"
        finally:
            config_path.unlink()

    def test_load_config_single_object(self):
        """Test loading config as single object"""
        config_obj = {
            "class": "SingleSelector",
            "params": {"value": 42}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_obj, f)
            config_path = Path(f.name)
        
        try:
            result = load_config(config_path)
            assert len(result) == 1
            assert result[0]["class"] == "SingleSelector"
        finally:
            config_path.unlink()

    def test_load_config_file_not_exists(self):
        """Test loading non-existent config file"""
        config_path = Path("non_existent_config.json")
        
        with pytest.raises(SystemExit):
            load_config(config_path)


class TestLoadAppendixCodes:
    @patch("select_stock.Path")
    def test_load_appendix_codes_success(self, mock_path, sample_appendix):
        """Test successful loading of appendix codes"""
        mock_file = mock_open(read_data=json.dumps(sample_appendix))
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.open = mock_file
        
        result = load_appendix_codes()
        
        assert result == ["000001", "000002", "600000", "600036"]

    @patch("select_stock.Path")
    def test_load_appendix_codes_file_not_exists(self, mock_path, caplog):
        """Test loading appendix when file doesn't exist"""
        mock_path.return_value.exists.return_value = False
        
        result = load_appendix_codes()
        
        assert result == []
        assert "appendix.json 文件不存在" in caplog.text

    @patch("select_stock.Path")
    def test_load_appendix_codes_invalid_json(self, mock_path, caplog):
        """Test loading appendix with invalid JSON"""
        mock_file = mock_open(read_data="invalid json")
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.open = mock_file
        
        result = load_appendix_codes()
        
        assert result == []
        assert "读取 appendix.json 失败" in caplog.text


class TestLoadHS300Codes:
    @patch("select_stock.importlib.import_module")
    def test_load_hs300_codes_success_sina(self, mock_import):
        """Test successful HS300 loading via sina API"""
        mock_ak = Mock()
        mock_df = pd.DataFrame({"code": ["000001", "000002", "600000"]})
        mock_ak.index_stock_cons_sina.return_value = mock_df
        mock_import.return_value = mock_ak
        
        result = load_hs300_codes()
        
        assert result == ["000001", "000002", "600000"]

    @patch("select_stock.importlib.import_module")
    def test_load_hs300_codes_fallback_csindex(self, mock_import):
        """Test HS300 loading fallback to csindex API"""
        mock_ak = Mock()
        mock_ak.index_stock_cons_sina.side_effect = Exception("API Error")
        mock_df = pd.DataFrame({"成分券代码": ["000001", "000002"]})
        mock_ak.index_stock_cons_csindex.return_value = mock_df
        mock_import.return_value = mock_ak
        
        result = load_hs300_codes()
        
        assert "000001" in result
        assert "000002" in result

    @patch("select_stock.importlib.import_module")
    def test_load_hs300_codes_backup_list(self, mock_import, caplog):
        """Test HS300 loading with backup list when APIs fail"""
        mock_ak = Mock()
        mock_ak.index_stock_cons_sina.side_effect = Exception("API Error")
        mock_ak.index_stock_cons_csindex.side_effect = Exception("API Error")
        mock_import.return_value = mock_ak
        
        result = load_hs300_codes()
        
        assert len(result) > 0
        assert "使用备选沪深300股票列表" in caplog.text

    @patch("select_stock.importlib.import_module")
    def test_load_hs300_codes_import_error(self, mock_import, caplog):
        """Test HS300 loading when akshare import fails"""
        mock_import.side_effect = ImportError("No module named 'akshare'")
        
        result = load_hs300_codes()
        
        assert len(result) > 0
        assert "使用备选沪深300股票列表" in caplog.text


class TestLoadA500Codes:
    @patch("select_stock.importlib.import_module")
    def test_load_a500_codes_success(self, mock_import):
        """Test successful A500 loading"""
        mock_ak = Mock()
        mock_df = pd.DataFrame({"code": ["002027", "002129", "002142"]})
        mock_ak.index_stock_cons_sina.return_value = mock_df
        mock_import.return_value = mock_ak
        
        result = load_a500_codes()
        
        assert result == ["002027", "002129", "002142"]


class TestLoadCombinedCodes:
    @patch("select_stock.load_hs300_codes")
    @patch("select_stock.load_a500_codes")
    @patch("select_stock.load_appendix_codes")
    def test_load_combined_codes(self, mock_appendix, mock_a500, mock_hs300):
        """Test loading combined codes with deduplication"""
        mock_hs300.return_value = ["000001", "000002", "600000"]
        mock_a500.return_value = ["002027", "002129", "000001"]  # 000001 is duplicate
        mock_appendix.return_value = ["600036", "000001"]  # 000001 is duplicate
        
        result = load_combined_codes()
        
        # Should be deduplicated
        assert len(result) == 5  # 000001, 000002, 600000, 002027, 002129, 600036
        assert "000001" in result
        assert "600036" in result


class TestInstantiateSelector:
    def test_instantiate_selector_missing_class(self):
        """Test selector instantiation with missing class field"""
        cfg = {"params": {"test": True}}
        
        with pytest.raises(ValueError, match="缺少 class 字段"):
            instantiate_selector(cfg)

    @patch("select_stock.importlib.import_module")
    def test_instantiate_selector_success(self, mock_import):
        """Test successful selector instantiation"""
        mock_module = Mock()
        mock_selector_class = Mock()
        mock_selector_instance = Mock()
        mock_selector_class.return_value = mock_selector_instance
        mock_module.TestSelector = mock_selector_class
        mock_import.return_value = mock_module
        
        cfg = {
            "class": "TestSelector",
            "alias": "测试",
            "params": {"threshold": 0.5}
        }
        
        alias, selector = instantiate_selector(cfg)
        
        assert alias == "测试"
        assert selector == mock_selector_instance
        mock_selector_class.assert_called_once_with(threshold=0.5)

    @patch("select_stock.importlib.import_module")
    def test_instantiate_selector_import_error(self, mock_import):
        """Test selector instantiation with import error"""
        mock_import.side_effect = ModuleNotFoundError("No module named 'Selector'")
        
        cfg = {"class": "TestSelector"}
        
        with pytest.raises(ImportError, match="无法加载 Selector.TestSelector"):
            instantiate_selector(cfg)


class TestMain:
    @patch("select_stock.sys.argv", ["select_stock.py", "--help"])
    def test_main_help(self):
        """Test main function with help argument"""
        with pytest.raises(SystemExit):
            main()

    @patch("select_stock.load_data")
    @patch("select_stock.load_config")
    @patch("select_stock.instantiate_selector")
    @patch("select_stock.Path")
    @patch("select_stock.sys.argv", ["select_stock.py", "--data-dir", "test_data"])
    def test_main_basic_flow(self, mock_path, mock_instantiate, mock_load_config, mock_load_data):
        """Test main function basic execution flow"""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.glob.return_value = [Mock(stem="000001"), Mock(stem="000002")]
        
        mock_data = {
            "000001": pd.DataFrame({
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "close": [10.0, 11.0]
            })
        }
        mock_load_data.return_value = mock_data
        
        mock_config = [{
            "class": "TestSelector",
            "activate": True,
            "params": {}
        }]
        mock_load_config.return_value = mock_config
        
        mock_selector = Mock()
        mock_selector.select.return_value = ["000001"]
        mock_instantiate.return_value = ("TestSelector", mock_selector)
        
        # Execute main
        main()
        
        # Verify calls
        mock_load_data.assert_called_once()
        mock_load_config.assert_called_once()
        mock_instantiate.assert_called_once()
        mock_selector.select.assert_called_once()

    @patch("select_stock.load_appendix_codes")
    @patch("select_stock.load_data")
    @patch("select_stock.load_config")
    @patch("select_stock.Path")
    @patch("select_stock.sys.argv", ["select_stock.py", "--only-appendix"])
    def test_main_only_appendix(self, mock_path, mock_load_config, mock_load_data, mock_load_appendix):
        """Test main function with --only-appendix flag"""
        mock_path.return_value.exists.return_value = True
        mock_load_appendix.return_value = ["000001", "000002"]
        mock_load_data.return_value = {}
        mock_load_config.return_value = []
        
        with pytest.raises(SystemExit):  # Will exit due to no data
            main()
        
        mock_load_appendix.assert_called_once()

    @patch("select_stock.Path")
    @patch("select_stock.sys.argv", ["select_stock.py", "--data-dir", "nonexistent"])
    def test_main_data_dir_not_exists(self, mock_path):
        """Test main function when data directory doesn't exist"""
        mock_path.return_value.exists.return_value = False
        
        with pytest.raises(SystemExit):
            main()
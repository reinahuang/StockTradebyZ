import pytest
import pandas as pd
import json
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime
import select_stock

# Import the module to test using absolute import


class TestLoadData:
    def test_load_data_existing_files(self, tmp_path):
        """Test loading data when CSV files exist"""
        # Create test CSV files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create sample CSV data
        csv_content = "date,open,high,low,close,volume\n2024-01-01,100,110,95,105,1000\n2024-01-02,105,115,100,110,1100\n"
        (data_dir / "000001.csv").write_text(csv_content)
        (data_dir / "000002.csv").write_text(csv_content)
        
        codes = ["000001", "000002"]
        result = select_stock.load_data(data_dir, codes)
        
        assert len(result) == 2
        assert "000001" in result
        assert "000002" in result
        assert isinstance(result["000001"], pd.DataFrame)
        assert len(result["000001"]) == 2
        assert "date" in result["000001"].columns

    def test_load_data_missing_files(self, tmp_path):
        """Test loading data when some CSV files don't exist"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create only one file
        csv_content = "date,open,high,low,close,volume\n2024-01-01,100,110,95,105,1000\n"
        (data_dir / "000001.csv").write_text(csv_content)
        
        codes = ["000001", "000999"]  # 000999 doesn't exist
        result = select_stock.load_data(data_dir, codes)
        
        assert len(result) == 1
        assert "000001" in result
        assert "000999" not in result

    def test_load_data_empty_codes(self, tmp_path):
        """Test loading data with empty codes list"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        result = select_stock.load_data(data_dir, [])
        assert len(result) == 0

    def test_load_data_sorts_by_date(self, tmp_path):
        """Test that data is sorted by date"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create CSV with unsorted dates
        csv_content = "date,close\n2024-01-03,103\n2024-01-01,101\n2024-01-02,102\n"
        (data_dir / "000001.csv").write_text(csv_content)
        
        result = select_stock.load_data(data_dir, ["000001"])
        df = result["000001"]
        
        assert df.iloc[0]["date"] < df.iloc[1]["date"] < df.iloc[2]["date"]


class TestLoadConfig:
    def test_load_config_list_format(self, tmp_path):
        """Test loading config in list format"""
        config_file = tmp_path / "config.json"
        config_data = [{"class": "TestSelector", "alias": "test"}]
        config_file.write_text(json.dumps(config_data))
        
        result = select_stock.load_config(config_file)
        assert len(result) == 1
        assert result[0]["class"] == "TestSelector"

    def test_load_config_dict_with_selectors(self, tmp_path):
        """Test loading config in dict format with selectors key"""
        config_file = tmp_path / "config.json"
        config_data = {"selectors": [{"class": "BBIKDJSelector", "alias": "少妇战法测试"}]}
        config_file.write_text(json.dumps(config_data))
        
        result = select_stock.load_config(config_file)
        assert len(result) == 1
        assert result[0]["class"] == "BBIKDJSelector"
        assert result[0]["alias"] == "少妇战法测试"

    def test_load_config_single_object(self, tmp_path):
        """Test loading config as single object"""
        config_file = tmp_path / "config.json"
        config_data = {"class": "TestSelector", "alias": "test"}
        config_file.write_text(json.dumps(config_data))
        
        result = select_stock.load_config(config_file)
        assert len(result) == 1
        assert result[0]["class"] == "TestSelector"

    def test_load_config_file_not_exists(self, tmp_path):
        """Test loading config when file doesn't exist"""
        config_file = tmp_path / "nonexistent.json"
        
        with pytest.raises(SystemExit) as exc_info:
            select_stock.load_config(config_file)
        assert exc_info.value.code == 1

    def test_load_config_empty_selectors(self, tmp_path):
        """Test loading config with empty selectors"""
        config_file = tmp_path / "config.json"
        config_data = {"selectors": []}
        config_file.write_text(json.dumps(config_data))
        
        with pytest.raises(SystemExit) as exc_info:
            select_stock.load_config(config_file)
        assert exc_info.value.code == 1


class TestLoadAppendixCodes:
    def test_load_appendix_codes_success(self, tmp_path, monkeypatch):
        """Test successful loading of appendix codes"""
        # Change working directory to temp path
        monkeypatch.chdir(tmp_path)
        
        appendix_file = tmp_path / "appendix.json"
        appendix_data = {"data": ["000001", "000002", "600000"]}
        appendix_file.write_text(json.dumps(appendix_data))
        
        result = select_stock.load_appendix_codes()
        assert len(result) == 3
        assert "000001" in result
        assert "000002" in result
        assert "600000" in result

    def test_load_appendix_codes_file_not_exists(self, tmp_path, monkeypatch):
        """Test loading appendix codes when file doesn't exist"""
        monkeypatch.chdir(tmp_path)
        
        result = select_stock.load_appendix_codes()
        assert result == []

    def test_load_appendix_codes_invalid_json(self, tmp_path, monkeypatch):
        """Test loading appendix codes with invalid JSON"""
        monkeypatch.chdir(tmp_path)
        
        appendix_file = tmp_path / "appendix.json"
        appendix_file.write_text("invalid json")
        
        result = select_stock.load_appendix_codes()
        assert result == []

    def test_load_appendix_codes_missing_data_key(self, tmp_path, monkeypatch):
        """Test loading appendix codes when data key is missing"""
        monkeypatch.chdir(tmp_path)
        
        appendix_file = tmp_path / "appendix.json"
        appendix_data = {"other_key": ["000001"]}
        appendix_file.write_text(json.dumps(appendix_data))
        
        result = select_stock.load_appendix_codes()
        assert result == []


class TestLoadHS300Codes:
    @patch('builtins.__import__')
    def test_load_hs300_codes_success_sina(self, mock_import):
        """Test successful loading of HS300 codes via sina API"""
        mock_ak = Mock()
        mock_df = pd.DataFrame({"code": ["000001", "000002", "600000"]})
        mock_ak.index_stock_cons_sina.return_value = mock_df
        
        def import_side_effect(name, *args, **kwargs):
            if name == "akshare":
                return mock_ak
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        result = select_stock.load_hs300_codes()
        assert len(result) == 3
        assert "000001" in result
        mock_ak.index_stock_cons_sina.assert_called_once_with(symbol='000300')

    @patch('builtins.__import__')
    def test_load_hs300_codes_sina_fails_csindex_succeeds(self, mock_import):
        """Test fallback from sina to csindex API"""
        mock_ak = Mock()
        mock_ak.index_stock_cons_sina.side_effect = Exception("Sina API failed")
        mock_df = pd.DataFrame({"成分券代码": ["000001", "000002"]})
        mock_ak.index_stock_cons_csindex.return_value = mock_df
        
        def import_side_effect(name, *args, **kwargs):
            if name == "akshare":
                return mock_ak
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        result = select_stock.load_hs300_codes()
        assert len(result) == 2
        assert "000001" in result

    @patch('builtins.__import__')
    def test_load_hs300_codes_all_apis_fail(self, mock_import):
        """Test fallback to backup codes when all APIs fail"""
        mock_ak = Mock()
        mock_ak.index_stock_cons_sina.side_effect = Exception("API Error")
        mock_ak.index_stock_cons_csindex.side_effect = Exception("API Error")
        
        def import_side_effect(name, *args, **kwargs):
            if name == "akshare":
                return mock_ak
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        result = select_stock.load_hs300_codes()
        assert len(result) > 0  # Should return backup codes
        assert "000001" in result

    @patch('builtins.__import__')
    def test_load_hs300_codes_no_akshare(self, mock_import):
        """Test fallback when akshare is not available"""
        def import_side_effect(name, *args, **kwargs):
            if name == "akshare":
                raise ImportError("No module named 'akshare'")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        result = select_stock.load_hs300_codes()
        assert len(result) > 0  # Should return backup codes
        assert "000001" in result


class TestLoadA500Codes:
    @patch('builtins.__import__')
    def test_load_a500_codes_success(self, mock_import):
        """Test successful loading of A500 codes"""
        mock_ak = Mock()
        mock_df = pd.DataFrame({"code": ["002027", "002129", "002142"]})
        mock_ak.index_stock_cons_sina.return_value = mock_df
        
        def import_side_effect(name, *args, **kwargs):
            if name == "akshare":
                return mock_ak
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        result = select_stock.load_a500_codes()
        assert len(result) == 3
        assert "002027" in result
        mock_ak.index_stock_cons_sina.assert_called_once_with(symbol='000905')

    @patch('builtins.__import__')
    def test_load_a500_codes_fallback(self, mock_import):
        """Test fallback to backup codes when akshare fails"""
        def import_side_effect(name, *args, **kwargs):
            if name == "akshare":
                raise ImportError("akshare not available")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        result = select_stock.load_a500_codes()
        assert len(result) > 0  # Should return backup codes
        assert "002027" in result


class TestLoadCombinedCodes:
    @patch('select_stock.load_hs300_codes')
    @patch('select_stock.load_a500_codes')
    @patch('select_stock.load_appendix_codes')
    def test_load_combined_codes(self, mock_appendix, mock_a500, mock_hs300):
        """Test loading combined codes from all sources"""
        mock_hs300.return_value = ["000001", "000002", "600000"]
        mock_a500.return_value = ["002027", "002129", "000001"]  # 000001 overlaps
        mock_appendix.return_value = ["300001", "000001"]  # 000001 overlaps
        
        result = select_stock.load_combined_codes()
        # Should be deduplicated
        unique_codes = set(result)
        assert len(unique_codes) == 5  # 000001, 000002, 600000, 002027, 002129, 300001
        assert "000001" in result
        assert "002027" in result
        assert "300001" in result

    @patch('select_stock.load_hs300_codes')
    @patch('select_stock.load_a500_codes')
    @patch('select_stock.load_appendix_codes')
    def test_load_combined_codes_empty_sources(self, mock_appendix, mock_a500, mock_hs300):
        """Test loading combined codes when some sources are empty"""
        mock_hs300.return_value = ["000001", "000002"]
        mock_a500.return_value = []
        mock_appendix.return_value = []
        
        result = select_stock.load_combined_codes()
        assert len(result) == 2
        assert "000001" in result
        assert "000002" in result


class TestInstantiateSelector:
    def test_instantiate_selector_success(self):
        """Test successful selector instantiation"""
        cfg = {
            "class": "BBIKDJSelector",
            "alias": "少妇战法测试",
            "params": {"j_threshold": 1, "bbi_q_threshold": 0.1}
        }
        
        mock_module = Mock()
        mock_class = Mock()
        mock_module.BBIKDJSelector = mock_class
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        
        with patch('select_stock.importlib.import_module', return_value=mock_module):
            alias, selector = select_stock.instantiate_selector(cfg)
            
            assert alias == "少妇战法测试"
            assert selector == mock_instance
            mock_class.assert_called_once_with(j_threshold=1, bbi_q_threshold=0.1)

    def test_instantiate_selector_no_alias(self):
        """Test selector instantiation without alias"""
        cfg = {
            "class": "TestSelector",
            "params": {}
        }
        
        mock_module = Mock()
        mock_class = Mock()
        mock_module.TestSelector = mock_class
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        
        with patch('select_stock.importlib.import_module', return_value=mock_module):
            alias, selector = select_stock.instantiate_selector(cfg)
            
            assert alias == "TestSelector"  # Should use class name as alias

    def test_instantiate_selector_missing_class(self):
        """Test selector instantiation with missing class field"""
        cfg = {"alias": "test_alias"}
        
        with pytest.raises(ValueError, match="缺少 class 字段"):
            select_stock.instantiate_selector(cfg)

    def test_instantiate_selector_module_not_found(self):
        """Test selector instantiation with module not found"""
        cfg = {"class": "NonExistentSelector"}
        
        with patch('select_stock.importlib.import_module', side_effect=ModuleNotFoundError("Module not found")):
            with pytest.raises(ImportError, match="无法加载 Selector.NonExistentSelector"):
                select_stock.instantiate_selector(cfg)

    def test_instantiate_selector_class_not_found(self):
        """Test selector instantiation with class not found in module"""
        cfg = {"class": "NonExistentSelector"}
        
        mock_module = Mock()
        del mock_module.NonExistentSelector  # Simulate missing attribute
        
        with patch('select_stock.importlib.import_module', return_value=mock_module):
            with patch('builtins.getattr', side_effect=AttributeError("No such attribute")):
                with pytest.raises(ImportError, match="无法加载 Selector.NonExistentSelector"):
                    select_stock.instantiate_selector(cfg)


class TestMain:
    @patch('select_stock.argparse.ArgumentParser')
    @patch('select_stock.Path.exists')
    @patch('select_stock.load_data')
    @patch('select_stock.load_config')
    @patch('select_stock.instantiate_selector')
    def test_main_basic_flow(self, mock_instantiate, mock_load_config, mock_load_data, mock_exists, mock_parser):
        """Test basic main function flow"""
        # Mock arguments
        mock_args = Mock()
        mock_args.data_dir = "./data"
        mock_args.config = "./configs.json"
        mock_args.date = "2024-01-01"
        mock_args.tickers = "000001,000002"
        mock_args.only_appendix = False
        mock_args.combined_appendix = False
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance
        
        # Mock Path.exists
        mock_exists.return_value = True
        
        # Mock data loading
        mock_data = {
            "000001": pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "close": [100]}),
            "000002": pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "close": [200]})
        }
        mock_load_data.return_value = mock_data
        
        # Mock config loading
        mock_config = [{"class": "TestSelector", "activate": True}]
        mock_load_config.return_value = mock_config
        
        # Mock selector
        mock_selector = Mock()
        mock_selector.select.return_value = ["000001"]
        mock_instantiate.return_value = ("TestSelector", mock_selector)
        
        select_stock.main()
        
        mock_load_data.assert_called_once()
        mock_load_config.assert_called_once()
        mock_instantiate.assert_called_once()
        mock_selector.select.assert_called_once()

    @patch('select_stock.argparse.ArgumentParser')
    @patch('select_stock.Path.exists')
    @patch('select_stock.sys.exit')
    def test_main_data_dir_not_exists(self, mock_exit, mock_exists, mock_parser):
        """Test main function when data directory doesn't exist"""
        mock_args = Mock()
        mock_args.data_dir = "./nonexistent"
        mock_parser.return_value.parse_args.return_value = mock_args
        
        mock_exists.return_value = False
        
        select_stock.main()
        mock_exit.assert_called_with(1)

    @patch('select_stock.argparse.ArgumentParser')
    @patch('select_stock.Path.exists')
    @patch('select_stock.load_appendix_codes')
    @patch('select_stock.load_data')
    @patch('select_stock.sys.exit')
    def test_main_only_appendix_flag(self, mock_exit, mock_load_data, mock_load_appendix, mock_exists, mock_parser):
        """Test main function with only-appendix flag"""
        mock_args = Mock()
        mock_args.data_dir = "./data"
        mock_args.tickers = "all"
        mock_args.only_appendix = True
        mock_args.combined_appendix = False
        
        mock_parser.return_value.parse_args.return_value = mock_args
        mock_exists.return_value = True
        mock_load_appendix.return_value = []  # Empty appendix
        mock_load_data.return_value = {}
        
        select_stock.main()
        mock_load_appendix.assert_called_once()
        mock_exit.assert_called_with(1)  # Should exit due to empty stock pool

    @patch('select_stock.argparse.ArgumentParser')
    @patch('select_stock.Path.exists')
    @patch('select_stock.load_combined_codes')
    @patch('select_stock.load_data')
    @patch('select_stock.sys.exit')
    def test_main_combined_appendix_flag(self, mock_exit, mock_load_data, mock_load_combined, mock_exists, mock_parser):
        """Test main function with combined-appendix flag"""
        mock_args = Mock()
        mock_args.data_dir = "./data"
        mock_args.tickers = "all"
        mock_args.only_appendix = False
        mock_args.combined_appendix = True
        
        mock_parser.return_value.parse_args.return_value = mock_args
        mock_exists.return_value = True
        mock_load_combined.return_value = []  # Empty combined
        mock_load_data.return_value = {}
        
        select_stock.main()
        mock_load_combined.assert_called_once()
        mock_exit.assert_called_with(1)  # Should exit due to empty stock pool


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for testing"""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1100, 1200, 1300, 1400]
    })


class TestIntegration:
    def test_load_data_integration(self, tmp_path, sample_dataframe):
        """Integration test for load_data function"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Save sample data as CSV
        csv_file = data_dir / "000001.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        
        result = select_stock.load_data(data_dir, ["000001"])
        
        assert "000001" in result
        loaded_df = result["000001"]
        assert len(loaded_df) == 5
        assert "date" in loaded_df.columns
        assert loaded_df["date"].dtype == "datetime64[ns]"
        
    def test_config_with_selector_example(self, tmp_path):
        """Test config loading with real selector example"""
        config_file = tmp_path / "configs.json"
        config_data = {
            "selectors": [{
                "class": "BBIKDJSelector",
                "alias": "少妇战法测试",
                "activate": True,
                "params": {
                    "j_threshold": 1,
                    "bbi_q_threshold": 0.1,
                    "j_q_threshold": 0.1
                }
            }]
        }
        config_file.write_text(json.dumps(config_data, ensure_ascii=False))
        
        result = select_stock.load_config(config_file)
        assert len(result) == 1
        selector_cfg = result[0]
        assert selector_cfg["class"] == "BBIKDJSelector"
        assert selector_cfg["alias"] == "少妇战法测试"
        assert selector_cfg["activate"] is True
        assert selector_cfg["params"]["j_threshold"] == 1
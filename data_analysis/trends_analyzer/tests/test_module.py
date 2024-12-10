import io
import unittest
import pandas as pd
from unittest.mock import patch
import warnings
import sys
import os
from statsmodels.tsa.stattools import acf
from trends_analyzer.trends_analyzer.analyzer_modules import TimeSeriesAnalyzer


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """
    Тестовый класс для проверки функциональности класса TimeSeriesAnalyzer.
    """

    @patch('trends_analyzer.trends_analyzer.analyzer_modules.TrendReq')
    def setUp(self, MockTrendReq):
        """
        Устанавливает тестовые данные и мок объекта TrendReq перед выполнением каждого теста.

        Args:
            MockTrendReq: Мок-класс для имитации поведения TrendReq из библиотеки pytrends.
        """
        # mock_pytrends = MockTrendReq.return_value
        # mock_data = pd.DataFrame({
        #     'minecraft': [10, 12, 15, 13, 18, 20, 25, 22, 28, 30, 35, 30, 28, 26, 25, 24,
        #                   10, 12, 15, 13, 18, 20, 25, 22, 28, 30, 35, 30, 28, 26, 25, 24]
        # })
        # mock_data['date'] = pd.date_range(start='2018-02-01', periods=len(mock_data),
        #                                   freq='D')
        # mock_data.set_index('date', inplace=True)
        # mock_pytrends.interest_over_time.return_value = mock_data

        # self.analyzer_module = TimeSeriesAnalyzer(keyword='minecraft',
        #                                           timeframe='today 5-y')
        self.data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.analyzer_module = TimeSeriesAnalyzer(keyword="test_keyword", timeframe="test_timeframe", data=self.data)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.held_output = io.StringIO()
        sys.stdout = self.held_output
    def test_calculate_moving_average(self):
        """
        Тестирует метод calculate_moving_average на корректность вычисления
        скользящего среднего.
        """
        result = self.analyzer_module.calculate_moving_average(window_size=7)
        expected_result = self.analyzer_module.data.rolling(window=7).mean()
        pd.testing.assert_series_equal(result, expected_result, check_names=False)

    def test_calculate_differential(self):
        """
        Тестирует метод calculate_differential на корректность вычисления
        дифференциала скользящего среднего.
        """
        result = self.analyzer_module.calculate_differential()
        expected_result = self.analyzer_module.moving_average_data.diff()
        pd.testing.assert_series_equal(result, expected_result, check_names=False)

    def test_calculate_auto_correlation(self):
        """
        Тестирует метод calculate_auto_correlation на корректность вычисления
        автокорреляции скользящего среднего.
        """
        result = self.analyzer_module.calculate_auto_correlation()

        valid_data = self.analyzer_module.moving_average_data.dropna()
        expected_result = acf(valid_data, nlags=len(valid_data) - 1)[1:]

        for lag, (res_val, exp_val) in enumerate(zip(result, expected_result), start=1):
            self.assertAlmostEqual(res_val, exp_val, places=5, msg=f"Failed at lag {lag}")

    def test_find_maxima(self):
        """
        Тестирует метод find_maxima на корректность нахождения локальных максимумов
        в скользящем среднем.
        """
        result = self.analyzer_module.find_maxima()
        expected_result = self.analyzer_module.moving_average_data[
            (self.analyzer_module.moving_average_data.shift(1)
             < self.analyzer_module.moving_average_data) &
            (self.analyzer_module.moving_average_data.shift(-1)
             < self.analyzer_module.moving_average_data)
            ]
        pd.testing.assert_series_equal(result, expected_result)

    def test_find_minima(self):
        """
        Тестирует метод find_minima на корректность нахождения локальных минимумов
        в скользящем среднем.
        """
        result = self.analyzer_module.find_minima()
        expected_result = self.analyzer_module.moving_average_data[
            (self.analyzer_module.moving_average_data.shift(1)
             > self.analyzer_module.moving_average_data) &
            (self.analyzer_module.moving_average_data.shift(-1)
             > self.analyzer_module.moving_average_data)
            ]
        pd.testing.assert_series_equal(result, expected_result)

    def test_save_to_dataframe(self):
        """
        Тестирует метод save_to_dataframe на корректность сохранения данных
        в DataFrame.

        Проверяет, что переданные данные корректно сохраняются с указанным именем столбца.
        """
        test_series = pd.Series([1, 2, 3, 4], name='Test Data')
        result = self.analyzer_module.save_to_dataframe(test_series, 'Test Column')
        expected_result = pd.DataFrame({'Test Column': test_series})
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_results(self):
        """
        Тестирует метод get_results на корректность получения всех результатов
        анализа.

        Проверяет, что все ожидаемые столбцы присутствуют в результате.
        """
        result = self.analyzer_module.get_results()
        self.assertIn('Moving Average', result.columns)
        self.assertIn('Differential', result.columns)
        self.assertIn('Auto correlation', result.columns)
        self.assertIn('Maxima', result.columns)
        self.assertIn('Minima', result.columns)


if __name__ == '__main__':
    unittest.main()

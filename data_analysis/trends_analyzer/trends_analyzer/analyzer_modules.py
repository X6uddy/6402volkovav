import pandas as pd
from typing import Any, Callable, Generator
from pytrends.request import TrendReq
from statsmodels.tsa.stattools import acf
from functools import wraps


def validate_args(func: Callable) -> Callable:
    """
    Декоратор для проверки корректности аргументов функции.

    Для функции calculate_moving_average проверяет, что window_size является положительным
    целым числом.
    Для функции calculate_auto_correlation проверяет, что lags является положительным
    целым числом.

    Args:
        func (Callable): Функция, к которой применяется декоратор.

    Returns:
        Callable: Обернутая функция.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        if func.__name__ == "calculate_moving_average":
            window_size = kwargs.get("window_size", 7)
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError(
                    "Аргумент 'window_size' должен быть положительным целым числом.")

        elif func.__name__ == "calculate_auto_correlation":
            lags = kwargs.get("lags", 40)
            if not isinstance(lags, int) or lags <= 0:
                raise ValueError(
                    "Аргумент 'lags' должен быть положительным целым числом.")

        result = func(*args, **kwargs)
        return result

    return wrapper


def generate_lags(n: int) -> Generator[int, None, None]:
    """
    Генератор для создания списка значений лагов от 1 до n.

    Args:
        n (int): Количество лагов.

    Yields:
        int: Значение лага от 1 до n.
    """
    for lag in range(1, n + 1):
        yield lag


class TimeSeriesAnalyzer:
    """
    Класс для анализа временного ряда по данным Google Trends.

    Методы позволяют вычислять скользящее среднее, дифференциал, автокорреляцию,
    а также находить максимумы и минимумы во временном ряде.

    Attributes:
        keyword (str): Ключевое слово для анализа.
        timeframe (str): Временной интервал данных в формате Google Trends.
        data (pd.Series): Временной ряд интереса по ключевому слову.
        moving_avg_data (pd.Series): Ряд скользящего среднего.
    """

    def __init__(self, keyword: str, timeframe: str) -> None:
        """
        Инициализирует TimeSeriesAnalyzer с заданным ключевым словом и временным
        интервалом.

        Args:
            keyword (str): Ключевое слово для анализа.
            timeframe (str): Временной интервал для данных
            (формат 'YYYY-MM-DD YYYY-MM-DD').
        """
        self.pytrends = TrendReq()
        self.keyword = keyword
        self.timeframe = timeframe
        self.data = self._load_data()
        self.moving_avg_data = self.calculate_moving_average()

    def _load_data(self) -> pd.Series:
        """
        Загружает данные из Google Trends по ключевому слову и возвращает временной ряд.

        Returns:
            pd.Series: Временной ряд с данными интереса по ключевому слову.
        """
        self.pytrends.build_payload([self.keyword], timeframe=self.timeframe)
        df = self.pytrends.interest_over_time()
        return df[self.keyword].dropna()

    @validate_args
    def calculate_moving_average(self, window_size: int = 7) -> pd.Series:
        """
        Вычисляет скользящее среднее по временным данным.

        Args:
            window_size (int, optional): Размер окна для скользящего среднего.
            По умолчанию 7.

        Returns:
            pd.Series: Временной ряд со значениями скользящего среднего.
        """
        if len(self.data) < window_size:
            return pd.Series(dtype=float)
        return self.data.rolling(window=window_size).mean()

    @validate_args
    def calculate_differential(self) -> pd.Series:
        """
        Вычисляет дифференциал от скользящего среднего для анализа изменений во времени.

        Returns:
            pd.Series: Временной ряд с дифференциалом скользящего среднего.
        """
        return self.moving_avg_data.diff()

    def calculate_auto_correlation(self, lags: int = None) -> pd.Series:
        """
        Вычисляет автокорреляцию скользящего среднего для заданного количества лагов.
        """
        if self.moving_avg_data.isnull().all():
            return pd.Series(
                [float('nan')] * len(self.moving_avg_data),
                index=self.moving_avg_data.index
            )

        valid_data = self.moving_avg_data.dropna()

        if lags is None:
            lags = len(valid_data) - 1

        if len(valid_data) < 2:
            return pd.Series(
                [float('nan')] * len(self.moving_avg_data),
                index=self.moving_avg_data.index
            )

        # Вычисление автокорреляции
        acf_values = acf(valid_data, nlags=lags)

        # Создание серии с NaN значениями для соответствия длине данных
        acf_series = pd.Series([float('nan')] * (len(self.moving_avg_data) - len(acf_values[1:])),
                               index=self.moving_avg_data.index[:len(self.moving_avg_data) - len(acf_values[1:])])

        # Соединение NaN значений и рассчитанных значений автокорреляции
        acf_series = pd.concat([acf_series, pd.Series(acf_values[1:], index=valid_data.index[:len(acf_values[1:])])])

        # Приведение индекса к полному индексу данных
        acf_series.index = self.moving_avg_data.index

        return acf_series

    @validate_args
    def find_maxima(self) -> pd.Series:
        """
        Находит локальные максимумы в скользящем среднем временного ряда.

        Returns:
            pd.Series: Временной ряд с локальными максимумами.
        """
        return self.moving_avg_data[
            (self.moving_avg_data.shift(1) < self.moving_avg_data) &
            (self.moving_avg_data.shift(-1) < self.moving_avg_data)
            ]

    @validate_args
    def find_minima(self) -> pd.Series:
        """
        Находит локальные минимумы в скользящем среднем временного ряда.

        Returns:
            pd.Series: Временной ряд с локальными минимумами.
        """
        return self.moving_avg_data[
            (self.moving_avg_data.shift(1) > self.moving_avg_data) &
            (self.moving_avg_data.shift(-1) > self.moving_avg_data)
            ]

    @validate_args
    def save_to_dataframe(self, result: pd.Series, name: str) -> pd.DataFrame:
        """
        Сохраняет переданный результат в DataFrame с указанным именем столбца.

        Args:
            result (pd.Series): Данные для сохранения.
            name (str): Имя столбца.

        Returns:
            pd.DataFrame: DataFrame с результатами.
        """
        return pd.DataFrame({name: result})

    def get_results(self) -> pd.DataFrame:
        """
        Получает и объединяет результаты всех вычислений в один DataFrame.

        Returns:
            pd.DataFrame: DataFrame со столбцами для скользящего среднего, дифференциала,
                          автокорреляции, максимумов и минимумов.
        """
        moving_avg = self.calculate_moving_average()
        differential = self.calculate_differential()
        auto_correlation = self.calculate_auto_correlation()
        maxima = self.find_maxima()
        minima = self.find_minima()

        results_df = pd.DataFrame({
            'Moving Average': moving_avg,
            'Differential': differential,
            'Auto correlation': auto_correlation,
            'Maxima': maxima,
            'Minima': minima,
        })

        return results_df


def execute_main():
    pass


if __name__ == '__main__':
    execute_main()
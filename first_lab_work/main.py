"""
Этот модуль выполняет вычисления и обрабатывает данные.
"""
import argparse
import math

import yaml


def read_config(filename: str) -> dict:
    """
    Считывает конфигурацию из файла

    Args:
        filename (str): Путь к файлу конфигурации

    Returns:
        dict: Содержимое файла конфигурации
    """
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)

    return config


def calculate_y(x_value: float, param_a: float, param_b: float, param_c: float) -> float:
    """
    Вычисляет значение функции y(x_value).

    Args:
        x_value (float): Входное значение
        param_a (float): Параметр a
        param_b (float): Параметр b
        param_c (float): Параметр c

    Returns:
        float: Вычисленное значение y
    """
    return 2 * x_value + (param_a * math.sin(param_b * x_value + param_c) ** 2) / (3 + x_value)


def main() -> None:
    """
    Основная функция для выполнения вычислений
    """
    parser = argparse.ArgumentParser(description='Calculate function y(x_value).')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--n0', type=float, help='Starting value of x_value')
    parser.add_argument('--step', type=float, help='Step size')
    parser.add_argument('--nk', type=float, help='Ending value of x_value')
    parser.add_argument('--a', type=float, help='Parameter a')
    parser.add_argument('--b', type=float, help='Parameter b')
    parser.add_argument('--c', type=float, help='Parameter c')
    args = parser.parse_args()
    config = read_config(args.config)
    n0 = args.n0 if args.n0 is not None else config['n0']
    step = args.step if args.step is not None else config['h']
    nk = args.nk if args.nk is not None else config['nk']
    param_a = args.a if args.a is not None else config['a']
    param_b = args.b if args.b is not None else config['b']
    param_c = args.c if args.c is not None else config['c']
    with open('output_results.txt', 'w') as results_file:
        x_value = n0
        while x_value <= nk:
            y_value = calculate_y(x_value, param_a, param_b, param_c)
            results_file.write(f"x_value = {x_value:.2f}, y = {y_value:.4f}\n")
            x_value += step


if __name__ == '__main__':
    main()

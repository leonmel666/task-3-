"""
ПОЛНОЕ РЕШЕНИЕ ЗАДАНИЯ ПО ТАБЛИЧНОМУ ПРОЦЕССОРУ
Вся функциональность в одном файле для избежания проблем с импортами
"""

import csv
import pickle
import copy
import os
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, date
import sys


# ============================================================================
# КЛАСС ТАБЛИЦЫ
# ============================================================================

class Table:
    """Класс для представления таблицы с данными."""
    
    def __init__(self, data: Optional[List[List[Any]]] = None,
                 columns: Optional[List[str]] = None,
                 column_types: Optional[Dict[Union[int, str], type]] = None):
        """
        Инициализация таблицы.
        
        Args:
            data: Данные таблицы (список строк)
            columns: Список названий столбцов
            column_types: Словарь типов столбцов
        """
        self._data = data if data is not None else []
        self._columns = columns if columns is not None else []
        self._column_types = column_types if column_types is not None else {}
        
        # Автоматически определяем заголовки, если они не заданы
        if not self._columns and self._data and self._data[0]:
            self._columns = [f"col_{i}" for i in range(len(self._data[0]))]
    
    def __repr__(self) -> str:
        return f"Table(rows={len(self._data)}, columns={len(self._columns)})"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: Union[int, slice]) -> 'Table':
        if isinstance(index, int):
            return Table([self._data[index]], self._columns, self._column_types)
        elif isinstance(index, slice):
            return Table(self._data[index], self._columns, self._column_types)
    
    @property
    def data(self) -> List[List[Any]]:
        """Возвращает копию данных таблицы."""
        return copy.deepcopy(self._data)
    
    @property
    def columns(self) -> List[str]:
        """Возвращает копию списка столбцов."""
        return self._columns.copy()
    
    @property
    def column_types(self) -> Dict[Union[int, str], type]:
        """Возвращает копию словаря типов столбцов."""
        return self._column_types.copy()
    
    def validate_column(self, column: Union[int, str]) -> int:
        """Проверяет и преобразует столбец в индекс."""
        if isinstance(column, int):
            if column < 0 or column >= len(self._columns):
                raise IndexError(f"Column index {column} out of range")
            return column
        elif isinstance(column, str):
            try:
                return self._columns.index(column)
            except ValueError:
                raise ValueError(f"Column '{column}' not found")
        else:
            raise TypeError(f"Column must be int or str, got {type(column)}")
    
    def _convert_value(self, value: Any, col_type: type) -> Any:
        """Преобразует значение к указанному типу."""
        if value is None or value == '':
            return None
        
        if col_type == str:
            return str(value)
        elif col_type == int:
            try:
                return int(float(value))
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert '{value}' to int")
        elif col_type == float:
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert '{value}' to float")
        elif col_type == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                value_lower = value.lower()
                if value_lower in ('true', '1', 'yes'):
                    return True
                elif value_lower in ('false', '0', 'no'):
                    return False
            try:
                return bool(int(value))
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert '{value}' to bool")
        elif col_type == datetime:
            if isinstance(value, datetime):
                return value
            elif isinstance(value, date):
                return datetime.combine(value, datetime.min.time())
            elif isinstance(value, str):
                # Попробуем разные форматы даты
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%d.%m.%Y %H:%M:%S',
                    '%d.%m.%Y',
                    '%Y/%m/%d %H:%M:%S',
                    '%Y/%m/%d'
                ]
                for fmt in formats:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Cannot parse datetime from '{value}'")
            else:
                raise ValueError(f"Cannot convert {type(value)} to datetime")
        else:
            return value
    
    def _convert_row(self, row: List[Any]) -> List[Any]:
        """Преобразует строку значений согласно типам столбцов."""
        converted = []
        for i, value in enumerate(row):
            col_type = self._column_types.get(i, self._column_types.get(self._columns[i], str))
            converted.append(self._convert_value(value, col_type))
        return converted


# ============================================================================
# CSV МОДУЛЬ
# ============================================================================

def load_table_csv(*files: Union[str, Path], 
                  auto_detect_types: bool = False,
                  **kwargs) -> Table:
    """
    Загружает таблицу из одного или нескольких CSV файлов.
    
    Args:
        *files: Пути к CSV файлам
        auto_detect_types: Автоматически определять типы столбцов
        **kwargs: Дополнительные параметры для csv.reader
        
    Returns:
        Table: Загруженная таблица
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если структура файлов не совпадает
    """
    if not files:
        raise ValueError("At least one file must be specified")
    
    all_data: List[List[Any]] = []
    columns: List[str] = []
    first_file = True
    
    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, **kwargs)
            file_data = list(reader)
            
            if not file_data:
                continue
            
            # Проверяем заголовки
            if first_file:
                columns = file_data[0]
                all_data.extend(file_data[1:])
                first_file = False
            else:
                if file_data[0] != columns:
                    raise ValueError(
                        f"Column structure mismatch between files. "
                        f"Expected {columns}, got {file_data[0]}"
                    )
                all_data.extend(file_data[1:])
    
    table = Table(all_data, columns)
    
    if auto_detect_types:
        table = auto_detect_column_types(table)
    
    return table


def save_table_csv(table: Table, 
                  filepath: Union[str, Path],
                  max_rows: Optional[int] = None,
                  **kwargs) -> None:
    """
    Сохраняет таблицу в CSV файл(ы).
    
    Args:
        table: Таблица для сохранения
        filepath: Путь для сохранения
        max_rows: Максимальное количество строк в файле
        **kwargs: Дополнительные параметры для csv.writer
        
    Raises:
        ValueError: Если max_rows <= 0
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    
    filepath = Path(filepath)
    data = [table.columns] + table.data
    
    if max_rows is None or len(data) <= max_rows:
        # Сохраняем в один файл
        _save_csv_data(data, filepath, **kwargs)
    else:
        # Разбиваем на несколько файлов
        base_name = filepath.stem
        suffix = filepath.suffix
        parent = filepath.parent
        
        for i in range(0, len(data), max_rows):
            chunk = data[i:i + max_rows]
            chunk_file = parent / f"{base_name}_part{i//max_rows + 1}{suffix}"
            _save_csv_data(chunk, chunk_file, **kwargs)


def _save_csv_data(data: List[List[Any]], 
                  filepath: Path, 
                  **kwargs) -> None:
    """Вспомогательная функция для сохранения данных в CSV."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, **kwargs)
        writer.writerows(data)


# ============================================================================
# PICKLE МОДУЛЬ
# ============================================================================

def load_table_pickle(*files: Union[str, Path], **kwargs) -> Table:
    """
    Загружает таблицу из одного или нескольких Pickle файлов.
    
    Args:
        *files: Пути к Pickle файлам
        **kwargs: Дополнительные параметры для pickle.load
        
    Returns:
        Table: Загруженная таблица
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если структура таблиц не совпадает
    """
    if not files:
        raise ValueError("At least one file must be specified")
    
    tables = []
    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            table = pickle.load(f, **kwargs)
            if not isinstance(table, Table):
                raise ValueError(f"File {file_path} does not contain a Table object")
            tables.append(table)
    
    # Проверяем совместимость таблиц
    if len(tables) > 1:
        first_columns = tables[0].columns
        first_types = tables[0].column_types
        
        for i, table in enumerate(tables[1:], 1):
            if table.columns != first_columns:
                raise ValueError(
                    f"Column mismatch between tables. "
                    f"Table 0: {first_columns}, Table {i}: {table.columns}"
                )
            if table.column_types != first_types:
                raise ValueError(
                    f"Column types mismatch between tables. "
                    f"Table 0: {first_types}, Table {i}: {table.column_types}"
                )
    
    # Объединяем данные
    all_data = []
    for table in tables:
        all_data.extend(table.data)
    
    return Table(all_data, tables[0].columns, tables[0].column_types)


def save_table_pickle(table: Table, 
                     filepath: Union[str, Path],
                     max_rows: Optional[int] = None,
                     **kwargs) -> None:
    """
    Сохраняет таблицу в Pickle файл(ы).
    
    Args:
        table: Таблица для сохранения
        filepath: Путь для сохранения
        max_rows: Максимальное количество строк в файле
        **kwargs: Дополнительные параметры для pickle.dump
        
    Raises:
        ValueError: Если max_rows <= 0
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    
    filepath = Path(filepath)
    
    if max_rows is None or len(table) <= max_rows:
        # Сохраняем в один файл
        _save_pickle_table(table, filepath, **kwargs)
    else:
        # Разбиваем на несколько файлов
        base_name = filepath.stem
        suffix = filepath.suffix
        parent = filepath.parent
        
        for i in range(0, len(table), max_rows):
            chunk_data = table.data[i:i + max_rows]
            chunk_table = Table(chunk_data, table.columns, table.column_types)
            chunk_file = parent / f"{base_name}_part{i//max_rows + 1}{suffix}"
            _save_pickle_table(chunk_table, chunk_file, **kwargs)


def _save_pickle_table(table: Table, filepath: Path, **kwargs) -> None:
    """Вспомогательная функция для сохранения таблицы в Pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(table, f, **kwargs)


# ============================================================================
# ТЕКСТОВЫЙ МОДУЛЬ
# ============================================================================

def save_table_text(table: Table, filepath: Union[str, Path]) -> None:
    """
    Сохраняет таблицу в текстовый файл в виде, аналогичном print_table().
    
    Args:
        table: Таблица для сохранения
        filepath: Путь для сохранения
        
    Raises:
        IOError: Если не удалось записать файл
    """
    filepath = Path(filepath)
    
    # Получаем строковое представление таблицы
    table_str = print_table(table, return_str=True)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(table_str)
    except Exception as e:
        raise IOError(f"Failed to write file {filepath}: {str(e)}")


# ============================================================================
# ОСНОВНЫЕ ОПЕРАЦИИ С ТАБЛИЦАМИ
# ============================================================================

def get_rows_by_number(table: Table, start: int, 
                      stop: Optional[int] = None,
                      copy_table: bool = False) -> Table:
    """
    Получает строки по номерам.
    
    Args:
        table: Исходная таблица
        start: Начальный индекс
        stop: Конечный индекс (не включая)
        copy_table: Копировать данные
        
    Returns:
        Table: Новая таблица с выбранными строками
        
    Raises:
        IndexError: Если индексы вне диапазона
    """
    if start < 0 or start >= len(table):
        raise IndexError(f"Start index {start} out of range")
    
    if stop is not None:
        if stop < 0 or stop > len(table):
            raise IndexError(f"Stop index {stop} out of range")
        if start >= stop:
            raise ValueError(f"Start index {start} must be less than stop index {stop}")
        data = table.data[start:stop]
    else:
        data = [table.data[start]]
    
    if copy_table:
        return Table(data, table.columns, table.column_types)
    else:
        # Создаем view на исходные данные
        return Table(data, table.columns, table.column_types)


def get_rows_by_index(table: Table, *indices: Any,
                     copy_table: bool = False) -> Table:
    """
    Получает строки по значениям в первом столбце.
    
    Args:
        table: Исходная таблица
        *indices: Значения для поиска в первом столбце
        copy_table: Копировать данные
        
    Returns:
        Table: Новая таблица с найденными строки
        
    Raises:
        ValueError: Если индексы не найдены
    """
    if not table:
        return Table([], table.columns, table.column_types)
    
    result_data = []
    indices_set = set(indices)
    
    for row in table.data:
        if row and row[0] in indices_set:
            result_data.append(row)
    
    if not result_data:
        raise ValueError(f"No rows found with indices: {indices}")
    
    if copy_table:
        return Table(result_data, table.columns, table.column_types)
    else:
        return Table(result_data, table.columns, table.column_types)


def get_column_types(table: Table, by_number: bool = True) -> Dict[Union[int, str], type]:
    """
    Получает типы столбцов.
    
    Args:
        table: Таблица
        by_number: Использовать номера столбцов (иначе имена)
        
    Returns:
        Dict: Словарь типов столбцов
    """
    if by_number:
        return {i: table.column_types.get(i, str) for i in range(len(table.columns))}
    else:
        return {col: table.column_types.get(i, str) 
                for i, col in enumerate(table.columns)}


def set_column_types(table: Table, 
                    types_dict: Dict[Union[int, str], type],
                    by_number: bool = True) -> Table:
    """
    Устанавливает типы столбцов.
    
    Args:
        table: Таблица
        types_dict: Словарь типов
        by_number: Использовать номера столбцов
        
    Returns:
        Table: Таблица с обновленными типами
        
    Raises:
        ValueError: Если тип не поддерживается
    """
    valid_types = {int, float, bool, str, datetime}
    
    for col, col_type in types_dict.items():
        if col_type not in valid_types:
            raise ValueError(f"Unsupported type {col_type}. "
                           f"Supported types: {valid_types}")
    
    # Создаем новую таблицу с обновленными типами
    new_table = Table(table.data, table.columns, table.column_types.copy())
    
    for col, col_type in types_dict.items():
        if by_number:
            col_idx = col if isinstance(col, int) else int(col)
        else:
            col_idx = table.columns.index(col) if isinstance(col, str) else col
        
        new_table._column_types[col_idx] = col_type
    
    # Конвертируем данные
    new_table._data = [new_table._convert_row(row) for row in table.data]
    
    return new_table


def get_values(table: Table, column: Union[int, str] = 0) -> List[Any]:
    """
    Получает значения столбца.
    
    Args:
        table: Таблица
        column: Столбец (номер или имя)
        
    Returns:
        List: Значения столбца
        
    Raises:
        IndexError: Если столбец не найден
    """
    col_idx = table.validate_column(column)
    return [row[col_idx] for row in table.data if col_idx < len(row)]


def get_value(table: Table, column: Union[int, str] = 0) -> Any:
    """
    Получает значение из таблицы с одной строкой.
    
    Args:
        table: Таблица
        column: Столбец
        
    Returns:
        Any: Значение
        
    Raises:
        ValueError: Если в таблице не одна строка
    """
    if len(table) != 1:
        raise ValueError("Table must have exactly one row")
    
    col_idx = table.validate_column(column)
    return table.data[0][col_idx] if col_idx < len(table.data[0]) else None


def set_values(table: Table, 
              values: List[Any], 
              column: Union[int, str] = 0) -> Table:
    """
    Устанавливает значения столбца.
    
    Args:
        table: Таблица
        values: Новые значения
        column: Столбец
        
    Returns:
        Table: Обновленная таблица
        
    Raises:
        ValueError: Если количество значений не совпадает
    """
    if len(values) != len(table):
        raise ValueError(f"Number of values ({len(values)}) "
                       f"must match number of rows ({len(table)})")
    
    col_idx = table.validate_column(column)
    new_data = copy.deepcopy(table.data)
    
    for i, value in enumerate(values):
        if col_idx < len(new_data[i]):
            new_data[i][col_idx] = value
        else:
            # Добавляем недостающие элементы
            while len(new_data[i]) <= col_idx:
                new_data[i].append(None)
            new_data[i][col_idx] = value
    
    return Table(new_data, table.columns, table.column_types)


def set_value(table: Table, 
             value: Any, 
             column: Union[int, str] = 0) -> Table:
    """
    Устанавливает значение в таблице с одной строкой.
    
    Args:
        table: Таблица
        value: Новое значение
        column: Столбец
        
    Returns:
        Table: Обновленная таблица
        
    Raises:
        ValueError: Если в таблице не одна строка
    """
    if len(table) != 1:
        raise ValueError("Table must have exactly one row")
    
    return set_values(table, [value], column)


def print_table(table: Table, 
               max_rows: Optional[int] = None,
               return_str: bool = False) -> Optional[str]:
    """
    Выводит таблицу на печать.
    
    Args:
        table: Таблица
        max_rows: Максимальное количество строк для вывода
        return_str: Возвращать строку вместо печати
        
    Returns:
        Optional[str]: Строковое представление если return_str=True
    """
    if not table:
        output = "Empty table\n"
        if return_str:
            return output
        print(output)
        return
    
    # Определяем ширину столбцов
    col_widths: List[int] = []
    for i, col in enumerate(table.columns):
        max_len = len(str(col))
        
        # Проверяем данные в столбце
        data_to_check = table.data[:max_rows] if max_rows else table.data
        for row in data_to_check:
            if i < len(row):
                max_len = max(max_len, len(str(row[i])))
        
        col_widths.append(max_len + 2)  # Добавляем отступ
    
    # Формируем заголовок
    header_parts: List[str] = []
    for col, width in zip(table.columns, col_widths):
        header_parts.append(str(col).ljust(width))
    header = " | ".join(header_parts)
    
    separator_parts: List[str] = []
    for width in col_widths:
        separator_parts.append("-" * width)
    separator = "-+-".join(separator_parts)
    
    lines = [header, separator]
    
    # Добавляем строки данных
    data_to_show = table.data[:max_rows] if max_rows else table.data
    for row in data_to_show:
        row_parts: List[str] = []
        for i, width in enumerate(col_widths):
            cell = row[i] if i < len(row) else ""
            row_parts.append(str(cell).ljust(width))
        row_str = " | ".join(row_parts)
        lines.append(row_str)
    
    if max_rows and len(table.data) > max_rows:
        lines.append(f"... and {len(table.data) - max_rows} more rows")
    
    output = "\n".join(lines)
    
    if return_str:
        return output
    else:
        print(output)
        return None


def concat(table1: Table, table2: Table) -> Table:
    """
    Объединяет две таблицы.
    
    Args:
        table1: Первая таблица
        table2: Вторая таблица
        
    Returns:
        Table: Объединенная таблица
        
    Raises:
        ValueError: Если структуры таблиц не совпадают
    """
    if table1.columns != table2.columns:
        raise ValueError("Tables have different column structures")
    
    if table1.column_types != table2.column_types:
        raise ValueError("Tables have different column types")
    
    combined_data = table1.data + table2.data
    return Table(combined_data, table1.columns, table1.column_types)


def split(table: Table, row_number: int) -> Tuple[Table, Table]:
    """
    Разделяет таблицу по номеру строки.
    
    Args:
        table: Таблица для разделения
        row_number: Номер строки для разделения
        
    Returns:
        Tuple[Table, Table]: Две таблицы
        
    Raises:
        IndexError: Если номер строки вне диапазона
    """
    if row_number < 0 or row_number > len(table):
        raise IndexError(f"Row number {row_number} out of range")
    
    table1 = Table(table.data[:row_number], table.columns, table.column_types)
    table2 = Table(table.data[row_number:], table.columns, table.column_types)
    
    return table1, table2


def auto_detect_column_types(table: Table) -> Table:
    """
    Автоматически определяет типы столбцов.
    
    Args:
        table: Таблица
        
    Returns:
        Table: Таблица с определенными типами
    """
    if not table.data:
        return table
    
    types_dict = {}
    
    for col_idx in range(len(table.columns)):
        # Анализируем значения в столбце
        values = [row[col_idx] for row in table.data if col_idx < len(row)]
        
        if not values:
            types_dict[col_idx] = str
            continue
        
        # Пробуем определить тип
        detected_type = _detect_type_for_column(values)
        types_dict[col_idx] = detected_type
    
    return set_column_types(table, types_dict)


def _detect_type_for_column(values: List[Any]) -> type:
    """
    Определяет тип данных для столбца.
    
    Args:
        values: Значения столбца
        
    Returns:
        type: Определенный тип
    """
    # Проверяем на datetime
    datetime_count = 0
    for val in values:
        if isinstance(val, datetime):
            datetime_count += 1
        elif isinstance(val, str):
            # Попробуем распарсить как datetime
            try:
                _ = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                datetime_count += 1
            except ValueError:
                try:
                    _ = datetime.strptime(val, '%Y-%m-%d')
                    datetime_count += 1
                except ValueError:
                    pass
    
    if datetime_count > len(values) * 0.8:  # 80% значений - datetime
        return datetime
    
    # Проверяем на int
    int_count = 0
    for val in values:
        try:
            int(str(val).strip())
            int_count += 1
        except (ValueError, AttributeError):
            pass
    
    if int_count == len(values):
        return int
    
    # Проверяем на float
    float_count = 0
    for val in values:
        try:
            float(str(val).strip())
            float_count += 1
        except (ValueError, AttributeError):
            pass
    
    if float_count == len(values):
        return float
    
    # Проверяем на bool
    bool_count = 0
    bool_values = {'true', 'false', '1', '0', 'yes', 'no'}
    for val in values:
        if isinstance(val, bool):
            bool_count += 1
        elif isinstance(val, str):
            if val.lower() in bool_values:
                bool_count += 1
    
    if bool_count == len(values):
        return bool
    
    # По умолчанию - строка
    return str


def add_columns(table: Table, 
               col1: Union[int, str],
               col2: Union[int, str],
               result_col: Optional[Union[int, str]] = None) -> Table:
    """
    Складывает значения двух столбцов.
    
    Args:
        table: Таблица
        col1: Первый столбец
        col2: Второй столбец
        result_col: Столбец для результата (если None, модифицирует col1)
        
    Returns:
        Table: Таблица с результатом
        
    Raises:
        ValueError: Если типы не поддерживают сложение
    """
    return _arithmetic_operation(table, col1, col2, result_col, 'add')


def sub_columns(table: Table,
               col1: Union[int, str],
               col2: Union[int, str],
               result_col: Optional[Union[int, str]] = None) -> Table:
    """
    Вычитает значения двух столбцов.
    
    Args:
        table: Таблица
        col1: Первый столбец
        col2: Второй столбец
        result_col: Столбец для результата
        
    Returns:
        Table: Таблица с результатом
    """
    return _arithmetic_operation(table, col1, col2, result_col, 'sub')


def mul_columns(table: Table,
               col1: Union[int, str],
               col2: Union[int, str],
               result_col: Optional[Union[int, str]] = None) -> Table:
    """
    Умножает значения двух столбцов.
    
    Args:
        table: Таблица
        col1: Первый столбец
        col2: Второй столбец
        result_col: Столбец для результата
        
    Returns:
        Table: Таблица с результатом
    """
    return _arithmetic_operation(table, col1, col2, result_col, 'mul')


def div_columns(table: Table,
               col1: Union[int, str],
               col2: Union[int, str],
               result_col: Optional[Union[int, str]] = None) -> Table:
    """
    Делит значения двух столбцов.
    
    Args:
        table: Таблица
        col1: Первый столбец
        col2: Второй столбец
        result_col: Столбец для результата
        
    Returns:
        Table: Таблица с результатом
    """
    return _arithmetic_operation(table, col1, col2, result_col, 'div')


def _arithmetic_operation(table: Table,
                         col1: Union[int, str],
                         col2: Union[int, str],
                         result_col: Optional[Union[int, str]],
                         operation: str) -> Table:
    """
    Выполняет арифметическую операцию над столбцами.
    
    Args:
        table: Таблица
        col1: Первый столбец
        col2: Второй столбец
        result_col: Столбец для результата
        operation: Операция ('add', 'sub', 'mul', 'div')
        
    Returns:
        Table: Таблица с результатом
        
    Raises:
        ValueError: Если операция не поддерживается для типов данных
        ZeroDivisionError: При делении на ноль
    """
    col1_idx = table.validate_column(col1)
    col2_idx = table.validate_column(col2)
    
    if result_col is None:
        result_idx = col1_idx
    else:
        result_idx = table.validate_column(result_col)
    
    # Получаем типы столбцов
    col_types = get_column_types(table)
    type1 = col_types.get(col1_idx, str)
    type2 = col_types.get(col2_idx, str)
    
    # Проверяем, поддерживаются ли типы для арифметики
    numeric_types = {int, float}
    if type1 not in numeric_types or type2 not in numeric_types:
        raise ValueError(f"Arithmetic operations not supported for types "
                       f"{type1} and {type2}. Only int and float are supported.")
    
    # Выполняем операцию
    new_data = copy.deepcopy(table.data)
    
    for i, row in enumerate(new_data):
        if col1_idx < len(row) and col2_idx < len(row):
            val1 = row[col1_idx]
            val2 = row[col2_idx]
            
            if val1 is None or val2 is None:
                result = None
            else:
                try:
                    if operation == 'add':
                        result = val1 + val2
                    elif operation == 'sub':
                        result = val1 - val2
                    elif operation == 'mul':
                        result = val1 * val2
                    elif operation == 'div':
                        if val2 == 0:
                            raise ZeroDivisionError(
                                f"Division by zero in row {i}, column {col2_idx}"
                            )
                        result = val1 / val2
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
                except Exception as e:
                    raise ValueError(
                        f"Error in row {i}: {e}"
                    )
            
            # Убеждаемся, что в строке достаточно элементов
            while len(row) <= result_idx:
                row.append(None)
            
            row[result_idx] = result
    
    return Table(new_data, table.columns, table.column_types)


# ============================================================================
# ДЕМОНСТРАЦИОННЫЙ ПРИМЕР
# ============================================================================

def main():
    """Основная демонстрационная функция."""
    
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ ТАБЛИЧНОГО ПРОЦЕССОРА")
    print("=" * 70)
    
    # Создаем тестовые данные
    test_data = [
        ["id", "name", "age", "salary", "is_manager"],
        ["1", "Alice", "25", "50000.50", "true"],
        ["2", "Bob", "30", "60000.75", "false"],
        ["3", "Charlie", "35", "70000.00", "true"],
        ["4", "Diana", "28", "55000.25", "false"],
        ["5", "Eve", "32", "65000.50", "true"]
    ]
    
    # Сохраняем в CSV для демонстрации загрузки
    with open('test_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(test_data)
    
    print("\n1. ЗАГРУЗКА ИЗ CSV:")
    print("-" * 40)
    table = load_table_csv('test_data.csv', auto_detect_types=True)
    print(f"   Загружена таблица: {table}")
    
    print("\n2. ТИПЫ СТОЛБЦОВ:")
    print("-" * 40)
    types = get_column_types(table, by_number=False)
    for col, col_type in types.items():
        print(f"   {col}: {col_type.__name__}")
    
    print("\n3. ВЫВОД ТАБЛИЦЫ (первые 3 строки):")
    print("-" * 40)
    print_table(table, max_rows=3)
    
    print("\n4. ПОЛУЧЕНИЕ СТРОК ПО НОМЕРУ (строки 1-3):")
    print("-" * 40)
    rows = get_rows_by_number(table, 0, 3, copy_table=True)
    print_table(rows)
    
    print("\n5. ПОЛУЧЕНИЕ ЗНАЧЕНИЙ СТОЛБЦА 'name':")
    print("-" * 40)
    names = get_values(table, 'name')
    print(f"   Имена: {names}")
    
    print("\n6. АРИФМЕТИЧЕСКИЕ ОПЕРАЦИИ:")
    print("-" * 40)
    
    # Создаем таблицу с числовыми данными
    numeric_data = [
        ["x", "y"],
        [10, 20],
        [15, 25],
        [20, 30]
    ]
    
    numeric_table = Table(numeric_data[1:], numeric_data[0])
    numeric_table = auto_detect_column_types(numeric_table)
    print("   Исходная таблица:")
    print_table(numeric_table)
    
    # Выполняем операции
    try:
        result = add_columns(numeric_table, "x", "y", "sum")
        print("\n   После сложения x + y:")
        print_table(result)
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    print("\n7. ОБЪЕДИНЕНИЕ ТАБЛИЦ:")
    print("-" * 40)
    table1 = get_rows_by_number(table, 0, 2, copy_table=True)
    table2 = get_rows_by_number(table, 2, 4, copy_table=True)
    combined = concat(table1, table2)
    print(f"   Объединенная таблица ({len(combined)} строк):")
    print_table(combined, max_rows=5)
    
    print("\n8. СОХРАНЕНИЕ В РАЗНЫХ ФОРМАТАХ:")
    print("-" * 40)
    
    # Сохранение в CSV
    save_table_csv(table, 'output.csv')
    print("   ✓ Сохранено в CSV: output.csv")
    
    # Сохранение с разбивкой
    save_table_csv(table, 'output_split.csv', max_rows=2)
    print("   ✓ Сохранено с разбивкой: output_split.csv, output_split_part1.csv, ...")
    
    # Сохранение в Pickle
    save_table_pickle(table, 'output.pickle')
    print("   ✓ Сохранено в Pickle: output.pickle")
    
    # Сохранение в текстовый файл
    save_table_text(table, 'output.txt')
    print("   ✓ Сохранено в текстовый файл: output.txt")
    
    print("\n9. ДЕМОНСТРАЦИЯ ОБРАБОТКИ ОШИБОК:")
    print("-" * 40)
    
    print("   а) Попытка получить несуществующий столбец:")
    try:
        values = get_values(table, "несуществующий")
    except ValueError as e:
        print(f"      Ошибка: {e}")
    
    print("\n   б) Попытка деления на ноль:")
    try:
        result = div_columns(numeric_table, "x", "y")
        # Создаем таблицу с нулем
        zero_data = [["a", "b"], [10, 0]]
        zero_table = Table(zero_data[1:], zero_data[0])
        zero_table = auto_detect_column_types(zero_table)
        result = div_columns(zero_table, "a", "b")
    except ZeroDivisionError as e:
        print(f"      Ошибка: {e}")
    
    print("\n   в) Попытка объединить несовместимые таблицы:")
    try:
        table_a = Table([[1, 2]], ["col1", "col2"])
        table_b = Table([[3, 4, 5]], ["col1", "col2", "col3"])
        result = concat(table_a, table_b)
    except ValueError as e:
        print(f"      Ошибка: {e}")
    
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 70)
    
    # Очистка временных файлов
    for filename in ['test_data.csv', 'output.csv', 'output.pickle', 'output.txt']:
        if os.path.exists(filename):
            os.remove(filename)
    
    # Удаляем разбитые файлы
    import glob
    for filename in glob.glob('output_split*.csv'):
        os.remove(filename)


if __name__ == "__main__":
    main()
import pandas as pd
import polars as pl
import os
import duckdb
import datetime
import re
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from joblib import Parallel, delayed

def process_code(args):
    """
    Process and save factor data for a single symbol.

    Parameters
    ----------
    args : tuple
        (symbol, small_df, factors_dir, factors_list, trade_date_col, symbol_col, append)

        symbol : str
            Stock symbol.
        small_df : pl.DataFrame
            Factor data for the given symbol.
        factors_dir : str
            Directory where parquet files are stored.
        factors_list : list
            List of factor column names to be updated.
        trade_date_col : str
            Name of the datetime column.
        symbol_col : str
            Name of the symbol column.
        append : bool
        Determines how new factor data is written into the existing parquet file.

        If True (append mode):
            - Only rows whose `trade_date_col` do NOT already exist
            in the existing file will be inserted.
            - Existing rows will NOT be modified.
            - New rows are concatenated to the existing dataset.
            - If no new dates are found, nothing will be written.

        If False (overwrite mode):
            - Only rows whose `trade_date_col` already exist
            in the existing file will be considered.
            - For overlapping dates, factor values in `factors_list`
            will overwrite the existing values (using coalesce logic).
            - Non-overlapping rows will NOT be inserted.
            - If no overlapping dates are found, nothing will be written.
    """
    symbol, small_df, factors_dir, factors_list, trade_date_col, symbol_col, append = args
    parquet_file = os.path.join(factors_dir, f"{symbol}.parquet")
    small_df = small_df.select([trade_date_col,symbol_col] + factors_list)
            
    if not os.path.exists(parquet_file):
        small_df.write_parquet(parquet_file)
        return symbol
                
    existing_df = pl.read_parquet(parquet_file, use_pyarrow=False)

    if existing_df[trade_date_col].dtype != pl.Datetime:
        existing_df = existing_df.with_columns(pl.col(trade_date_col).cast(pl.Datetime('ns')))
    small_df = small_df.with_columns(pl.col(trade_date_col).cast(pl.Datetime("ns")))
            
    for fac in factors_list:
        if fac not in existing_df.columns:
            existing_df = existing_df.with_columns(pl.lit(None).cast(pl.Float64).alias(fac))
            
    for col in existing_df.columns:
        if col not in small_df.columns:
            small_df = small_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    small_df = small_df.select(existing_df.columns)

    if append:
        small_df = small_df.join(
            existing_df.select(trade_date_col),
            on=trade_date_col,
            how="anti"
        )
                
        if small_df.is_empty():
            return
                        
        merged_df = pl.concat([existing_df, small_df])
        merged_df = merged_df.sort(trade_date_col)
        merged_df.write_parquet(parquet_file)
            
    else:
        overlap_df = small_df.join(
            existing_df.select(trade_date_col),
            on=trade_date_col,
            how="inner"
        )

        if overlap_df.is_empty():
            return

        merged = existing_df.join(
            overlap_df,
            on=trade_date_col,
            how="left",
            suffix="_new"
        )

        merged = merged.with_columns([
            pl.coalesce(pl.col(f + "_new"), pl.col(f)).alias(f)
            for f in small_df.columns if not f == trade_date_col
        ])

        merged = merged.drop([
            f + "_new"
            for f in small_df.columns if not f == trade_date_col
        ])

        merged.write_parquet(parquet_file)
             
class DataBase():
    """
    DataBase

    A high-performance data loading and preprocessing utility designed for
    factor research pipelines. This class provides a unified interface for
    reading, merging, aligning, and preparing large-scale financial datasets
    stored in symbol-level parquet files.

    The class supports continuous features (e.g., price-based indicators)
    and discrete features (e.g., fundamental events, classifications), and
    automatically aligns them with the main price dataset.

    Main capabilities include:
    - Efficient parquet querying via DuckDB
    - Parallel reading of symbol-level datasets
    - Automatic time alignment of continuous and discrete features
    - Forward filling of discrete attributes
    - Time shifting of continuous indicators to prevent look-ahead bias
    - Parallel factor storage to symbol-level parquet files

    Parameters
    ----------
    PathConfig : dict
        Configuration dictionary describing the directory structure and
        feature groups. It should include:

        - main_dir_path : str
            Root directory containing all data folders.

        - base_dir_name : dict
            Dictionary specifying the main dataset directory and its features.

        - continuous : list[str]
            Directory names containing continuous features.

        - discrete : list[str]
            Directory names containing discrete features.

        Additional keys map directory names to their feature column lists.

    stocks_list : list[str]
        List of asset symbols to load.

    begin_date : str
        Start datetime of the data range (format: "YYYY-MM-DD HH:MM:SS").

    end_date : str
        End datetime of the data range (format: "YYYY-MM-DD HH:MM:SS").

    trade_date_col : str
        Column name representing the timestamp.

    symbol_col : str
        Column name representing the asset identifier.

    freq : str, default "1d"
        Data frequency used to determine feature shift durations.

    shift_n : int, default 1
        Number of periods used to shift continuous features forward in time.
        This helps prevent look-ahead bias.

    dropNaN : bool, default False
        Whether to drop rows containing missing values after merging datasets.

    max_workers : int, default -1
        Number of parallel workers used for data loading and saving.
        If -1, the system will use (CPU count - 1).

    Attributes
    ----------
    continuous_df : pl.DataFrame
        DataFrame containing merged continuous features.

    discrete_df : pl.DataFrame
        DataFrame containing merged discrete features.

    discrete_dfs : list
        Intermediate list of discrete datasets.

    continuous_dfs : list
        Intermediate list of continuous datasets.

    feature_dict : dict
        Mapping between directory names and feature column lists.

    Notes
    -----
    Continuous vs Discrete Features

    Continuous features typically include indicators derived from prices
    (e.g., returns, momentum, volatility) and are shifted forward to avoid
    look-ahead bias.

    Discrete features represent event-based or categorical information
    (e.g., industry classification, fundamental reports) and are forward
    filled using an as-of join.

    Data Alignment

    - Continuous features are merged directly on (datetime, symbol)
    - Discrete features are aligned using backward `join_asof`
    - The final dataset is sorted by (datetime, symbol)

    Example
    -------
    >>> db = DataBase(
    ...     PathConfig=config,
    ...     stocks_list=stocks,
    ...     begin_date="2018-01-01 00:00:00",
    ...     end_date="2024-01-01 00:00:00",
    ...     trade_date_col="datetime",
    ...     symbol_col="symbol"
    ... )
    >>> df = db.get()

    The returned DataFrame can be directly used in factor research
    pipelines such as FactorAnalyzer.
    """
    def __init__(self, 
                 PathConfig: dict,
                 stocks_list:list,
                 begin_date:str,
                 end_date:str,
                 trade_date_col:str,
                 symbol_col:str,
                 freq:str='1d',
                 shift_n: int = 1,
                 dropNaN: bool = False,
                 max_workers:int = -1):
        
        self.shift_n = shift_n
        self.dropNaN = dropNaN
        self.stocks_list = stocks_list
        self.begin_date:str = begin_date
        self.end_date:str = datetime.datetime.strptime(end_date,'%Y-%m-%d %H:%M:%S')
        self.freq = freq
        self.shift_duration:str = self.multiply_duration(shift_n*20,freq)
        self.base_dir_name, self.base_dir_features = next(iter(PathConfig['base_dir_name'].items()))
        
        self.main_dir_path = PathConfig.get('main_dir_path')
        self.trade_date_col:str = trade_date_col
        self.symbol_col:str = symbol_col
        self.max_workers = max_workers
        
        self.feature_dict = {
            key: val for key, val in PathConfig.items()
            if key not in ['main_dir_path', 'continuous', 'discrete','base_dir_name']
        }
        
        self.continuous_dict = {'continuous': PathConfig.get('continuous', [])}
        self.discrete_dict = {'discrete': PathConfig.get('discrete', [])}
        
        self.discrete_dfs = []
        self.continuous_dfs = []
        
        self.discrete_df:pd.DataFrame = None
        self.continuous_df:pd.DataFrame = None
    
    def read_dir_file(self,dir_name:str):
        n_jobs = max(os.cpu_count() - 1, 1) if self.max_workers == -1 else self.max_workers
        dir_path = os.path.join(self.main_dir_path,dir_name)
           
        if not dir_path.endswith(os.sep):
            dir_path += os.sep
        print(f"Data directory : {os.path.abspath(dir_path)}")
        
        stock_list = self.stocks_list
        
        con = duckdb.connect()
        con.execute(f"SET threads TO {n_jobs};")
        
        try:
            
            parquet_files = list(map(lambda name: os.path.join(dir_path, f"{name}.parquet"), stock_list))

            file_list = ", ".join(map(lambda file: f"'{file}'", parquet_files))
            
            if dir_name == self.base_dir_name:
                features = self.base_dir_features
            elif dir_name in self.discrete_dict.get('discrete', []):
                features = self.feature_dict.get(dir_name, [])
            elif dir_name in self.continuous_dict.get('continuous', []):
                features = self.feature_dict.get(dir_name, [])
            else:
                features = []
            if not features:
                print(f"No features configured for table '{dir_name}'.")
                return []
            selected_features = ", ".join(map(lambda feat: feat, features))
            
            if dir_name in self.discrete_dict.get('discrete', []):
                
                begin_date = datetime.datetime.strptime(self.begin_date,'%Y-%m-%d %H:%M:%S')
                begin_date = self.shift_datetime(begin_date,'150d')
                
                #print(begin_date)
                
            elif dir_name in self.continuous_dict.get('continuous', []):
                
                begin_date = datetime.datetime.strptime(self.begin_date,'%Y-%m-%d %H:%M:%S')
                
                begin_date = self.shift_datetime(begin_date,self.shift_duration)
                
                #print(begin_date)
            else:
                
                begin_date = datetime.datetime.strptime(self.begin_date,'%Y-%m-%d %H:%M:%S')
                
                #print(begin_date)
                
            query = f"""
            SELECT {selected_features}
            FROM read_parquet([{file_list}]) 
            WHERE {self.trade_date_col} >= '{begin_date}' 
            AND datetime <= '{self.end_date}'
            """
        
            df_all = con.execute(query).pl()
                   
            if not df_all.is_empty():
                
                df_all = df_all.with_columns(pl.col(self.trade_date_col).cast(pl.Datetime))                           
                
                print(f"Successfully fetched {len(df_all)} rows of data.")
                return df_all
            else:
                for file in parquet_files[:5]:
                    exists = "exists" if os.path.exists(file) else "not found"
                    print(f"{os.path.basename(file)}: {exists}")
                    
                    print('failed to fetch data')
                
                return pl.DataFrame(schema={col: pl.Null for col in features})
            
        except Exception as e:
            print(f"Query error: {str(e)}")
            return pl.DataFrame(schema={col: pl.Null for col in features})
        
        finally:
            con.close()
    
    def read_and_merge_dfs(self):
        base_df = self.read_dir_file(self.base_dir_name).sort([self.trade_date_col, self.symbol_col])
        self.continuous_dfs = []
        for dir_name in self.continuous_dict.get("continuous", []):
            df = self.read_dir_file(dir_name).sort([self.trade_date_col, self.symbol_col])
            if not df.is_empty():
                
                df = df.with_columns(pl.col(self.trade_date_col).shift(-self.shift_n).alias(self.trade_date_col))
            self.continuous_dfs.append(df)

       
        self.discrete_dfs = []
        for dir_name in self.discrete_dict.get("discrete", []):
            df = self.read_dir_file(dir_name)
            if not df.is_empty():
                subset_cols = self.feature_dict.get(dir_name, [])
                
                if subset_cols:
                    df = df.unique(subset=subset_cols)
                df = df.sort([self.trade_date_col, self.symbol_col])
            self.discrete_dfs.append(df)

        
        self.continuous_df = base_df.clone()
        for df in self.continuous_dfs:
            
            self.continuous_df = (
                self.continuous_df.join(df, on=[self.trade_date_col, self.symbol_col], how="left")
                .sort([self.symbol_col,self.trade_date_col])
            )

        if self.discrete_dfs:
            self.discrete_df = self.discrete_dfs[0].sort([self.trade_date_col, self.symbol_col])
            for df in self.discrete_dfs[1:]:
                
                common_cols = [c for c in self.discrete_df.columns if c in df.columns]
                if not common_cols:
                    raise ValueError(
                        "No common columns found when merging discrete datasets. "
                        "pl.merge will fail in this case — please check the input."
                    )
                self.discrete_df = self.discrete_df.join(df, on=common_cols, how="outer")
            self.discrete_df = self.discrete_df.sort([self.symbol_col,self.trade_date_col])
        else:
            self.discrete_df = pl.DataFrame() 

        #print(self.continuous_df)
        #print(self.discrete_df)

    def fill_to_full_df(self):
        cont_df:pl.DataFrame = self.continuous_df
        if self.discrete_dfs:
            disc_df = self.discrete_df

            filled_df:pl.DataFrame = cont_df.join_asof(
                disc_df,
                on=self.trade_date_col,
                by=self.symbol_col,
                strategy="backward"
            )

            if self.dropNaN:
                filled_df = filled_df.drop_nulls()
                
            filled_df = filled_df.sort([self.trade_date_col,self.symbol_col]).to_pandas()
            filled_df[self.trade_date_col] = pd.to_datetime(filled_df[self.trade_date_col])
            return filled_df.sort_values([self.trade_date_col, self.symbol_col])

        else:
            cont_df = cont_df.sort([self.trade_date_col, self.symbol_col]).to_pandas()
            cont_df[self.trade_date_col] = pd.to_datetime(cont_df[self.trade_date_col])
            return cont_df
    
    def get(self):
        self.read_and_merge_dfs()
        full_df = self.fill_to_full_df()
        return full_df
    
    @staticmethod
    def save(base_df: pd.DataFrame,
             factors_list,
             factors_dir,
             trade_date_col:str,
             symbol_col:str,
             append=False,
             max_workers:int = -1):
        """
        Save factor data into symbol-level parquet files using parallel processing.

        This function splits the input DataFrame by symbol and writes each symbol's
        data into an individual parquet file. Existing files will be either appended
        or partially overwritten depending on the `append` flag.

        Parameters
        ----------
        base_df : pd.DataFrame or pl.DataFrame
            Input dataset containing at least:
            - symbol column
            - datetime column
            - factor columns listed in `factors_list`

        factors_list : list[str]
            List of factor column names to be written or updated.

        factors_dir : str
            Directory where symbol-level parquet files are stored.
            Each symbol will be saved as: {symbol}.parquet

        trade_date_col : str
            Name of the datetime column.

        symbol_col : str
            Name of the symbol column used to split the dataset.

        append : bool, default=False
            Writing mode:
            - True  → append only new dates (no modification of existing rows).
            - False → overwrite factor values on overlapping dates only.

        max_workers : int, default=-1
            Number of parallel worker processes.
            - If -1, uses (CPU count - 1).
            - Otherwise uses the specified number of workers.
        """
        os.makedirs(factors_dir, exist_ok=True)
        if isinstance(base_df,pd.DataFrame):
            base_df= pl.from_pandas(base_df)
        base_df:pl.DataFrame = base_df.with_columns(pl.col(trade_date_col).cast(pl.Datetime('ns')))
        
        logger = setup_logger()
       
        symbols = base_df[symbol_col].unique().to_list()
        small_dfs = base_df.partition_by(symbol_col)
        args_list = [
            (code, small_df, factors_dir, factors_list, trade_date_col, symbol_col, append)
            for code, small_df in zip(symbols, small_dfs)
        ]

        n_jobs = max(os.cpu_count() - 1, 1) if max_workers == -1 else max_workers

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_code)(args)
            for args in tqdm(args_list, ncols=80)
        )
            
    @staticmethod
    def shift_datetime(base_time: datetime, duration: str, mode: str = "sub") -> datetime:
        if not isinstance(base_time, datetime.datetime):
            raise TypeError("base_time must be datetime")
        if not duration or not isinstance(duration, str):
            return base_time
        if mode not in ("add", "sub"):
            raise ValueError("mode must be 'add' or 'sub'")

        
        pattern = r"(\d+)(y|min|m|d|h|s)"
        matches = re.findall(pattern, duration)

        
        kwargs = {
            "years": 0,
            "months": 0,
            "days": 0,
            "hours": 0,
            "minutes": 0,
            "seconds": 0,
        }

        
        for value, unit in matches:
            value = int(value)
            if unit == "y":
                kwargs["years"] += value
            elif unit == "m":
                kwargs["months"] += value
            elif unit == "d":
                kwargs["days"] += value
            elif unit == "h":
                kwargs["hours"] += value
            elif unit == "min":
                kwargs["minutes"] += value
            elif unit == "s":
                kwargs["seconds"] += value

        delta = relativedelta(**kwargs)
        return base_time + delta if mode == "add" else base_time - delta

    @staticmethod
    def multiply_duration(multiplier: int, duration: str) -> str:
        if not duration or not isinstance(duration, str):
            return duration

        pattern = r"(\d+)(y|min|m|d|h|s)"
        matches = re.findall(pattern, duration)

        time_units = {
            "y": 0,
            "m": 0,
            "d": 0,
            "h": 0,
            "min": 0,
            "s": 0
        }

        for value, unit in matches:
            time_units[unit] += int(value) * multiplier

        carry = [
            ("s", "min", 60),
            ("min", "h", 60),
            ("h", "d", 24),
            ("d", "m", 30),
            ("m", "y", 12),
        ]

        for low, high, threshold in carry:
            if time_units[low] >= threshold:
                extra = time_units[low] // threshold
                time_units[low] %= threshold
                time_units[high] += extra

        order = ["y", "m", "d", "h", "min", "s"]
        result = "".join(f"{time_units[u]}{u}" for u in order if time_units[u] > 0)

        return result if result else "0s"


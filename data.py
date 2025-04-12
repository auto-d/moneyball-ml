import pandas as pd
import numpy as np 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import pybaseball as pb
pb.cache.enable()

# Consolidation of all ingest, processing and cleaning for conventional 
# and statcast-based metrics 

players = pb.chadwick_register(save=True)

def find_player(id=None, name=None): 
    """
    Lookup a player name provided an ID, or first matching ID provided a tuple 
    of (<last name>, <first name>)
    """
    global players

    if id is not None:         
        return pb.playerid_reverse_lookup(id)
    else:
        last = None 
        first = None
        if type(name) == str: 
            # "last, first"
            last = name.split(sep=',')[0].strip() 
            first = name.split(sep=',')[1].strip()
        elif type(name) == tuple: 
            # (last, first)
            last = name[0]
            first = name[1]
        else:
            raise ValueError("Unknown type!")
        
        mlbid = None
        match = players[(players.name_last==last) & (players.name_first==first)]
        if len(match) >= 1: 
            mlbid = match.iloc[0].key_mlbam
        return mlbid

def idfg_to_mlb(id): 
    """
    Map a Fangraphs player ID to MLB player ID
    """
    global players

    mlbid = None
    match = players[players.key_fangraphs == id]
    if len(match) >= 1: 
        mlbid = match.iloc[0].key_mlbam
    
    return mlbid

def find_na(df): 
    """
    Track down any elusive NA values

    NOTE: this is a utilty function I wrote for the Kaggle comp
    """
    for col in df.columns: 
        na = len(df[df[col].isna()])
        if na > 0: 
            raise ValueError(f"{df}/{col} has {na} na values!") 

def scale(df, range=(0,1), omit=[]):
    """
    Scale a column into a given range, omitting one or more columns if provided 
    
    NOTE: this is a utilty function I wrote for the Kaggle comp
    """
    for column in df.columns: 
        if column not in omit: 
            df[column] = min_max_scale(df, column, (0,1))

    return df

def sum_by_index(df): 
    """
    Sum a (hopefully numeric) DF by it's index and return it

    NOTE: this is a utilty function I wrote for the Kaggle comp
    """
    squashed = df.groupby(df.index).sum()
    squashed.fillna(0, inplace=True)

    return squashed

def ordinal_feature(df, column): 
    """
    Scale a column in a DF to the provided range     

    NOTE: this is a utilty function I wrote for the Kaggle comp
    """
    encoder = OrdinalEncoder()
    encoder.fit(df[[column]])    
    ordinals = encoder.transform(df[[column]])
    return ordinals

def onehot_feature(df, column): 
    """
    One-hot encode a feature

    NOTE: based on a utilty function I wrote for the Kaggle comp
    """
    encoder = OneHotEncoder(handle_unknown='ignore') 
    encoder.fit(df[[column]])

    columns = []
    for feature in encoder.get_feature_names_out():         
        columns.append(feature.replace(' ', '_'))
                       
    new_df = pd.DataFrame(encoder.transform(df[[column]]).toarray(), columns=columns, index=df.index) 

    return new_df

def min_max_scale(df, column, range=(0,1)): 
    """
    Scale a column in a DF to the provided range     

    NOTE: this is a utilty function I wrote for the Kaggle comp
    """
    scaler = MinMaxScaler(feature_range=range)
    scaled = scaler.fit_transform(df[[column]]) 
    return scaled.transpose()[0]


def canonicalize_data(df, drop_pct=0.01, drop=[], onehot=[], ordinal=[], boolean=[]): 
    """
    One-stop data cleaning operation to streamline large dataframe ingest from MLB data sources. 

    This function: 
    1. Homogenizes numeric types
    2. Cleans small NA value rows where they less than the drop_pct threshold
    3. Drops columns indicated by the drop param
    4. One-hot encodes columns indicated by the onehot param 
    5. Ordinal encodes cols indicated by the ... ordinal param 
    6. Uses a present/absent heursitic to create a boolean column, yielding 0/1 for those indicated
       in the associated param 
    7. Drops columns it can't figure out how to convert, printing the unique values to aid in 
       updating the lists on future calls
    """

    dfc = pd.DataFrame(index=df.index)
    rows = len(df) 
    for type_, column in zip(df.dtypes, df.columns): 
        
        print(f"{column} (type={type_}):")

        # Drop where needed
        if column in drop: 
            print(' - dropping due to presence in drop list')
        elif column.endswith("_deprecated"): 
            print(" - dropping due to '_deprecated' suffix")
        else: 
            # Flag nans, and if they are less than a certain percentage of the DF, just drop the 
            # rows outright before attempting conversion.
            nans = df[column].isna().sum() 
            if nans != 0: 
                print(f" - {nans} nans present!")
                if nans/rows <= drop_pct: 
                    print(f" - <= {drop_pct*100}% of values, dropping affected rows!")
                    not_na = ~df[column].isna()
                    df = df[not_na]
                    dfc = dfc[not_na]
                    nans = 0 
                else: 
                    print(f" - >{drop_pct*100}% of values, ignoring!")
                    
            # Map each column over to a new DF, converting as needed to support downstream modeling
            if column in boolean: 
                dfc[column] = df[column].apply(lambda x: 0 if pd.isna(x) else 1) 
            elif pd.api.types.is_datetime64_ns_dtype(type_): 
                dfc[column] = pd.to_numeric(df[column])
                print(' - converted to int')
            elif pd.api.types.is_float(type_) or pd.api.types.is_float_dtype(type_) or type_ == np.float64: 
                if nans:
                    print(' - ❗️ WARNING: filling nans with 0!')
                    df[column] = df[column].fillna(0)
                dfc[column] = df[column].astype(np.float32)
                print(' - converted to float')
            elif pd.api.types.is_int64_dtype(type_) or type_ == np.int64:
                if nans:
                    print(' - ❗️ WARNING: filling nans with 0.0!')
                    df[column] = df[column].fillna(0)
                dfc[column] = df[column].astype(np.int32)
                print(' - converted to int')
            elif pd.api.types.is_string_dtype(type_) or type_ == str: 
                if column in onehot: 
                    onehot_df = onehot_feature(df, column)
                    dfc = pd.concat([dfc, onehot_df], axis=1)
                    print(f" - one-hot encoding") 
                    if nans: 
                        print(" - ❗️ WARNING nans converted to 0 for all new features")
                elif column in ordinal: 
                    # TODO: nans? 
                    dfc[column] = ordinal_feature(df, column)
                    print(f" - ordinal encoding!")                
                else: 
                    print(f" - feature not found in encode list, dropping!")
                    print(f" - feature values = {df[column].unique()}")
            else: 
                raise ValueError(f"Unknown type encountered ({type_}), can't process dataframe!")

    return dfc


def load_std_pitching(year, dir='data/'): 
    """
    Load conventional pitching statistics 
    See https://library.fangraphs.com/pitching/complete-list-pitching/
    """    
    file = dir + f"std_pitching_{year}.parquet"
    cdf = None
    try: 
        cdf = pd.read_parquet(file) 

    except FileNotFoundError: 
            
        df = pb.pitching_stats(year)

        drop_columns = [
            'Season',
            'Name',
            'Team',
            #'WAR', label 
            'RAR',
            'Dollars',
            'tERA',
            'xFIP',
            'WPA',
            '-WPA',
            '+WPA',
            'RE24',
            'REW',
            'pLI',
            'inLI',
            'gmLI',
            'exLI',
            'Pulls',
            'WPA/LI',
            'Clutch',
            'FB% 2',
            'FBv',
            'SL%',
            'SLv',
            'CT%',
            'CTv',
            'CB%',
            'CBv',
            'CH%',
            'CHv',
            'SF%',
            'SFv',
            'KN%',
            'KNv',
            'XX%',
            'PO%',
            'wFB',
            'wSL',
            'wCT',
            'wCB',
            'wCH',
            'wSF',
            'wKN',
            'wFB/C',
            'wSL/C',
            'wCT/C',
            'wCB/C',
            'wCH/C',
            'wSF/C',
            'wKN/C',
            'O-Swing%',
            'Z-Swing%',
            'Swing%',
            'O-Contact%',
            'Z-Contact%',
            'Contact%',
            'Zone%',
            'F-Strike%',
            'SwStr%',
            'HLD',
            'SD',
            'MD',
            'ERA-',
            'FIP-',
            'xFIP-',
            'E-F',
            'FA% (sc)',
            'FT% (sc)',
            'FC% (sc)',
            'FS% (sc)',
            'FO% (sc)',
            'SI% (sc)',
            'SL% (sc)',
            'CU% (sc)',
            'KC% (sc)',
            'EP% (sc)',
            'CH% (sc)',
            'SC% (sc)',
            'KN% (sc)',
            'UN% (sc)',
            'vFA (sc)',
            'vFT (sc)',
            'vFC (sc)',
            'vFS (sc)',
            'vFO (sc)',
            'vSI (sc)',
            'vSL (sc)',
            'vCU (sc)',
            'vKC (sc)',
            'vEP (sc)',
            'vCH (sc)',
            'vSC (sc)',
            'vKN (sc)',
            'FA-X (sc)',
            'FT-X (sc)',
            'FC-X (sc)',
            'FS-X (sc)',
            'FO-X (sc)',
            'SI-X (sc)',
            'SL-X (sc)',
            'CU-X (sc)',
            'KC-X (sc)',
            'EP-X (sc)',
            'CH-X (sc)',
            'SC-X (sc)',
            'KN-X (sc)',
            'FA-Z (sc)',
            'FT-Z (sc)',
            'FC-Z (sc)',
            'FS-Z (sc)',
            'FO-Z (sc)',
            'SI-Z (sc)',
            'SL-Z (sc)',
            'CU-Z (sc)',
            'KC-Z (sc)',
            'EP-Z (sc)',
            'CH-Z (sc)',
            'SC-Z (sc)',
            'KN-Z (sc)',
            'wFA (sc)',
            'wFT (sc)',
            'wFC (sc)',
            'wFS (sc)',
            'wFO (sc)',
            'wSI (sc)',
            'wSL (sc)',
            'wCU (sc)',
            'wKC (sc)',
            'wEP (sc)',
            'wCH (sc)',
            'wSC (sc)',
            'wKN (sc)',
            'wFA/C (sc)',
            'wFT/C (sc)',
            'wFC/C (sc)',
            'wFS/C (sc)',
            'wFO/C (sc)',
            'wSI/C (sc)',
            'wSL/C (sc)',
            'wCU/C (sc)',
            'wKC/C (sc)',
            'wEP/C (sc)',
            'wCH/C (sc)',
            'wSC/C (sc)',
            'wKN/C (sc)',
            'O-Swing% (sc)',
            'Z-Swing% (sc)',
            'Swing% (sc)',
            'O-Contact% (sc)',
            'Z-Contact% (sc)',
            'Contact% (sc)',
            'Zone% (sc)',
            'Pace',
            'RA9-WAR',
            'BIP-Wins',
            'LOB-Wins',
            'FDP-Wins',
            'Age Rng',
            'kwERA',
            'TTO%',
            'CH% (pi)',
            'CS% (pi)',
            'CU% (pi)',
            'FA% (pi)',
            'FC% (pi)',
            'FS% (pi)',
            'KN% (pi)',
            'SB% (pi)',
            'SI% (pi)',
            'SL% (pi)',
            'XX% (pi)',
            'vCH (pi)',
            'vCS (pi)',
            'vCU (pi)',
            'vFA (pi)',
            'vFC (pi)',
            'vFS (pi)',
            'vKN (pi)',
            'vSB (pi)',
            'vSI (pi)',
            'vSL (pi)',
            'vXX (pi)',
            'CH-X (pi)',
            'CS-X (pi)',
            'CU-X (pi)',
            'FA-X (pi)',
            'FC-X (pi)',
            'FS-X (pi)',
            'KN-X (pi)',
            'SB-X (pi)',
            'SI-X (pi)',
            'SL-X (pi)',
            'XX-X (pi)',
            'CH-Z (pi)',
            'CS-Z (pi)',
            'CU-Z (pi)',
            'FA-Z (pi)',
            'FC-Z (pi)',
            'FS-Z (pi)',
            'KN-Z (pi)',
            'SB-Z (pi)',
            'SI-Z (pi)',
            'SL-Z (pi)',
            'XX-Z (pi)',
            'wCH (pi)',
            'wCS (pi)',
            'wCU (pi)',
            'wFA (pi)',
            'wFC (pi)',
            'wFS (pi)',
            'wKN (pi)',
            'wSB (pi)',
            'wSI (pi)',
            'wSL (pi)',
            'wXX (pi)',
            'wCH/C (pi)',
            'wCS/C (pi)',
            'wCU/C (pi)',
            'wFA/C (pi)',
            'wFC/C (pi)',
            'wFS/C (pi)',
            'wKN/C (pi)',
            'wSB/C (pi)',
            'wSI/C (pi)',
            'wSL/C (pi)',
            'wXX/C (pi)',
            'O-Swing% (pi)',
            'Z-Swing% (pi)',
            'Swing% (pi)',
            'O-Contact% (pi)',
            'Z-Contact% (pi)',
            'Contact% (pi)',
            'Zone% (pi)',
            'Pace (pi)',
            'FRM',
            'K/9+',
            'BB/9+',
            'K/BB+',
            'H/9+',
            'HR/9+',
            'AVG+',
            'WHIP+',
            'BABIP+',
            'LOB%+',
            'K%+',
            'BB%+',
            'LD%+',
            'GB%+',
            'FB%+',
            'HR/FB%+',
            'Pull%+',
            'Cent%+',
            'Oppo%+',
            'Soft%+',
            'Med%+',
            'Hard%+',
            'EV',
            'LA',
            'Barrels',
            'Barrel%',
            'maxEV',
            'HardHit',
            'HardHit%',
            'Events',
            'CStr%',
            'CSW%',
            'xERA',
            'botERA',
            'botOvr CH',
            'botStf CH',
            'botCmd CH',
            'botOvr CU',
            'botStf CU',
            'botCmd CU',
            'botOvr FA',
            'botStf FA',
            'botCmd FA',
            'botOvr SI',
            'botStf SI',
            'botCmd SI',
            'botOvr SL',
            'botStf SL',
            'botCmd SL',
            'botOvr KC',
            'botStf KC',
            'botCmd KC',
            'botOvr FC',
            'botStf FC',
            'botCmd FC',
            'botOvr FS',
            'botStf FS',
            'botCmd FS',
            'botOvr',
            'botStf',
            'botCmd',
            'botxRV100',
            'Stf+ CH',
            'Loc+ CH',
            'Pit+ CH',
            'Stf+ CU',
            'Loc+ CU',
            'Pit+ CU',
            'Stf+ FA',
            'Loc+ FA',
            'Pit+ FA',
            'Stf+ SI',
            'Loc+ SI',
            'Pit+ SI',
            'Stf+ SL',
            'Loc+ SL',
            'Pit+ SL',
            'Stf+ KC',
            'Loc+ KC',
            'Pit+ KC',
            'Stf+ FC',
            'Loc+ FC',
            'Pit+ FC',
            'Stf+ FS',
            'Loc+ FS',
            'Pit+ FS',
            'Stuff+',
            'Location+',
            'Pitching+',
            'Stf+ FO',
            'Loc+ FO',
            'Pit+ FO',
        ]

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns)
        
        cdf.to_parquet(file)

    return cdf

def load_std_batting(year, dir='data/'): 
    """
    Load conventional batting statistics 
    """
    file = dir + f"std_batting_{year}.parquet"
    cdf = None
    try: 
        cdf = pd.read_parquet(file) 

    except FileNotFoundError: 
            
        df = pb.batting_stats(year, qual=1)
        
        drop_columns = [
            'Season',
            'Name',
            'Team',
            # We discard all post-hoc analysis that incorporates wins and statcast-derived insights
            'Bat',
            'Fld',
            'Rep',
            'Pos',
            'RAR',
            # 'WAR', ... lone exception, this is our label
            'WPA',
            '-WPA',
            '+WPA',
            'RE24',
            'REW',
            'pLI',
            'phLI',
            'PH',
            'WPA/LI',
            'Clutch',
            'Dol',
            'wFB',
            'wSL',
            'wCT',
            'wCB',
            'wCH',
            'wSF',
            'wKN',
            'wFB/C',
            'wSL/C',
            'wCT/C',
            'wCB/C',
            'wCH/C',
            'wSF/C',
            'wKN/C',
            'Barrels',
            'Barrel%',
            'maxEV',
            'HardHit',
            'HardHit%',
            'Events',
            'O-Swing%',
            'Z-Swing%',
            'Swing%',
            'O-Contact%',
            'Z-Contact%',
            'Contact%',
            'Zone%',
            'F-Strike%',
            'SwStr%',
            'BsR',
            'Pace',
            'Def',
            'wSB',
            'UBR',
            'Age Rng',
            'Off',
            'Lg',
            'wGDP',
            'Pull%+',   
            'Cent%+',
            'Oppo%+',
            'Soft%+',
            'Med%+',
            'Hard%+',
            'EV',
            'LA',
            'CStr%',
            'CSW%',
            'xBA',
            'xSLG',
            'xwOBA',
            'wRAA',
            'L-WAR',
            'FB% (Pitch)',
            'FBv',
            'SL%',
            'SLv',
            'CT%',
            'CTv',
            'CB%',
            'CBv',
            'CH%',
            'CHv',
            'SF%',
            'SFv',
            'KN%',
            'KNv',
            'XX%',
            'PO%',
            'wRC',
            'Spd',
            'wRC+',
            'FA% (sc)',
            'FT% (sc)',
            'FC% (sc)',
            'FS% (sc)',
            'FO% (sc)',
            'SI% (sc)',
            'SL% (sc)',
            'CU% (sc)',
            'KC% (sc)',
            'EP% (sc)',
            'CH% (sc)',
            'SC% (sc)',
            'KN% (sc)',
            'UN% (sc)',
            'vFA (sc)',
            'vFT (sc)',
            'vFC (sc)',
            'vFS (sc)',
            'vFO (sc)',
            'vSI (sc)',
            'vSL (sc)',
            'vCU (sc)',
            'vKC (sc)',
            'vEP (sc)',
            'vCH (sc)',
            'vSC (sc)',
            'vKN (sc)',
            'FA-X (sc)',
            'FT-X (sc)',
            'FC-X (sc)',
            'FS-X (sc)',
            'FO-X (sc)',
            'SI-X (sc)',
            'SL-X (sc)',
            'CU-X (sc)',
            'KC-X (sc)',
            'EP-X (sc)',
            'CH-X (sc)',
            'SC-X (sc)',
            'KN-X (sc)',
            'FA-Z (sc)',
            'FT-Z (sc)',
            'FC-Z (sc)',
            'FS-Z (sc)',
            'FO-Z (sc)',
            'SI-Z (sc)',
            'SL-Z (sc)',
            'CU-Z (sc)',
            'KC-Z (sc)',
            'EP-Z (sc)',
            'CH-Z (sc)',
            'SC-Z (sc)',
            'KN-Z (sc)',
            'wFA (sc)',
            'wFT (sc)',
            'wFC (sc)',
            'wFS (sc)',
            'wFO (sc)',
            'wSI (sc)',
            'wSL (sc)',
            'wCU (sc)',
            'wKC (sc)',
            'wEP (sc)',
            'wCH (sc)',
            'wSC (sc)',
            'wKN (sc)',
            'wFA/C (sc)',
            'wFT/C (sc)',
            'wFC/C (sc)',
            'wFS/C (sc)',
            'wFO/C (sc)',
            'wSI/C (sc)',
            'wSL/C (sc)',
            'wCU/C (sc)',
            'wKC/C (sc)',
            'wEP/C (sc)',
            'wCH/C (sc)',
            'wSC/C (sc)',
            'wKN/C (sc)',
            'O-Swing% (sc)',
            'Z-Swing% (sc)',
            'Swing% (sc)',
            'O-Contact% (sc)',
            'Z-Contact% (sc)',
            'Contact% (sc)',
            'Zone% (sc)',
            'CH% (pi)',
            'CS% (pi)',
            'CU% (pi)',
            'FA% (pi)',
            'FC% (pi)',
            'FS% (pi)',
            'KN% (pi)',
            'SB% (pi)',
            'SI% (pi)',
            'SL% (pi)',
            'XX% (pi)',
            'vCH (pi)',
            'vCS (pi)',
            'vCU (pi)',
            'vFA (pi)',
            'vFC (pi)',
            'vFS (pi)',
            'vKN (pi)',
            'vSB (pi)',
            'vSI (pi)',
            'vSL (pi)',
            'vXX (pi)',
            'CH-X (pi)',
            'CS-X (pi)',
            'CU-X (pi)',
            'FA-X (pi)',
            'FC-X (pi)',
            'FS-X (pi)',
            'KN-X (pi)',
            'SB-X (pi)',
            'SI-X (pi)',
            'SL-X (pi)',
            'XX-X (pi)',
            'CH-Z (pi)',
            'CS-Z (pi)',
            'CU-Z (pi)',
            'FA-Z (pi)',
            'FC-Z (pi)',
            'FS-Z (pi)',
            'KN-Z (pi)',
            'SB-Z (pi)',
            'SI-Z (pi)',
            'SL-Z (pi)',
            'XX-Z (pi)',
            'wCH (pi)',
            'wCS (pi)',
            'wCU (pi)',
            'wFA (pi)',
            'wFC (pi)',
            'wFS (pi)',
            'wKN (pi)',
            'wSB (pi)',
            'wSI (pi)',
            'wSL (pi)',
            'wXX (pi)',
            'wCH/C (pi)',
            'wCS/C (pi)',
            'wCU/C (pi)',
            'wFA/C (pi)',
            'wFC/C (pi)',
            'wFS/C (pi)',
            'wKN/C (pi)',
            'wSB/C (pi)',
            'wSI/C (pi)',
            'wSL/C (pi)',
            'wXX/C (pi)',
            'O-Swing% (pi)',
            'Z-Swing% (pi)',
            'Swing% (pi)',
            'O-Contact% (pi)',
            'Z-Contact% (pi)',
            'Contact% (pi)',
            'Zone% (pi)',
            'Pace (pi)',
        ]

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns)
        
        cdf.to_parquet(file)

    return cdf

def load_sc_base(year, dir): 
    """
    Load and clean pitching/hitting data, pulling from the Internet if we haven't already processed 
    and saved the cleaned data. 
    """
    file = dir + f"sc_{year}.parquet"
    cdf = None
    try: 
        cdf = pd.read_parquet(file) 

    except FileNotFoundError: 

        df = pb.statcast(start_dt=f"{year}-4-1", end_dt=f"{year}-9-30")

        drop_columns = [
            'player_name', 
            'spin_dir', 
            'spin_rate_deprecated', 
            'break_angle_deprecated', 
            'break_length_deprecated', 
            'tfs_deprecated', 
            'tfs_zulu_deprecated', 
            'des', 
            'game_type',
            'home_team', # these are both useful, but we need to engineer a feature perhaps to suggest whether 
            'away_team', # the player in question is home or away, and in what park ... 
            'type', 
            'hit_location', # poorly reported for pitchers
            'game_year', 
            'umpire', 
            'hc_x', # we have the sc x,y coords 
            'hc_y',
            'sv_id',
            'last_name, first_name'
            ]
        onehot_columns = [
            'pitch_type', 
            'events', 
            'description', 
            'stand', 
            'p_throws', 
            'bb_type',
            ]
        ordinal_columns = []
        bool_columns = [
            'on_3b',
            'on_2b',
            'on_1b',
        ]

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns, 
            onehot=onehot_columns, 
            ordinal=ordinal_columns, 
            boolean=bool_columns)
        
        cdf.to_parquet(file)

    return cdf

def load_sc_pitching_velo(year, dir): 
    """
    Load and clean pitching batted ball data
    """
    file = dir + f"sc_pitching_velo_{year}.parquet"
    df = None
    try: 
        cdf = pd.read_parquet(file) 

    except FileNotFoundError: 

        df = pb.statcast_pitcher_exitvelo_barrels(year, minBBE=1)

        drop_columns = [
            'last_name, first_name'
            ]

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns)
        
        cdf.to_parquet(file)

    return cdf

def load_sc_pitching_spin(year, dir): 
    """
    Load and clean pitching spin data
    """
    # Manual pull from baseball savant
    file = dir + f"active_spin_{year}.csv"
    df = None
    try: 
        df = pd.read_csv(file) 

        onehot_columns = [ 
            'pitch_hand', 
        ]
        drop_columns = [
            'last_name, first_name',
        ]

        df['player_id'] = df['last_name, first_name'].apply(lambda x: find_player(name=x))

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns, 
            onehot=onehot_columns)

    except FileNotFoundError as e: 
        raise e

    return cdf

def load_sc_pitching_mvmt(year, dir): 
    """
    Load and clean pitching movement data
    """
    # Manual pull from baseball savant
    file = dir + f"pitch_movement_{year}.csv"
    df = None
    try: 
        df = pd.read_csv(file)

        onehot_columns = [ 
            'pitch_hand', 
            'pitch_type' # redundant, in base pitching data but see if they are aligned

        ]
        drop_columns = [
            'year',
            'team_name'
            'team_name_abbrev',  
            'last_name, first_name', 
            'pitch_type_name', 
            'league_break_z'
            ]

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns, 
            onehot=onehot_columns)

    except FileNotFoundError as e: 
        raise e

    return cdf

def load_sc_pitching(year, dir): 
    """
    Load and clean statcast pitching data
    """
    sc_df = load_sc_base(year, dir)    

    df_pitcher = sc_df.groupby('pitcher').agg({
        'pitch_type_CH' : ['sum'], 
        'pitch_type_CS' : ['sum'], 
        'pitch_type_CU' : ['sum'], 
        'pitch_type_EP' : ['sum'],
        'pitch_type_FA' : ['sum'], 
        'pitch_type_FC' : ['sum'], 
        'pitch_type_FF' : ['sum'], 
        'pitch_type_FO' : ['sum'], 
        'pitch_type_FS' : ['sum'], 
        'pitch_type_KC' : ['sum'], 
        'pitch_type_KN' : ['sum'], 
        'pitch_type_PO' : ['sum'], 
        'pitch_type_SC' : ['sum'], 
        'pitch_type_SI' : ['sum'], 
        'pitch_type_SL' : ['sum'], 
        'pitch_type_ST' : ['sum'], 
        'pitch_type_SV' : ['sum'], 
        #'game_date', we don't have temporal data for other statcast metrics, have to retreat to season-level data :(
        'release_speed' : ['mean'], 
        'release_pos_x' : ['mean'], 
        'release_pos_z' : ['mean'], 
        #'batter', # interesting but would require a one-hot encoding for all batters and associated sparseness
        #'pitcher', # group feature
        'events_catcher_interf' : ['sum'], 
        'events_double' : ['sum'], 
        'events_double_play' : ['sum'], 
        'events_field_error' : ['sum'], 
        'events_field_out' : ['sum'], 
        'events_fielders_choice' : ['sum'], 
        'events_fielders_choice_out' : ['sum'], 
        'events_force_out' : ['sum'], 
        'events_grounded_into_double_play' : ['sum'], 
        'events_hit_by_pitch' : ['sum'], 
        'events_home_run' : ['sum'], 
        'events_sac_bunt' : ['sum'], 
        'events_sac_fly' : ['sum'], 
        'events_sac_fly_double_play' : ['sum'], 
        'events_single' : ['sum'], 
        'events_strikeout' : ['sum'], 
        'events_strikeout_double_play' : ['sum'], 
        'events_triple' : ['sum'], 
        'events_triple_play' : ['sum'], 
        'events_truncated_pa' : ['sum'], 
        'events_walk' : ['sum'], 
        #'events_None', 
        'description_ball' : ['sum'], 
        'description_blocked_ball' : ['sum'], 
        'description_bunt_foul_tip' : ['sum'], 
        'description_called_strike' : ['sum'], 
        'description_foul' : ['sum'], 
        'description_foul_bunt' : ['sum'], 
        'description_foul_tip' : ['sum'], 
        'description_hit_by_pitch' : ['sum'], 
        'description_hit_into_play' : ['sum'], 
        'description_missed_bunt' : ['sum'], 
        'description_pitchout' : ['sum'], 
        'description_swinging_strike' : ['sum'], 
        'description_swinging_strike_blocked' : ['sum'], 
        #'zone'
        })

    # groupby creates a column multindex to convey the aggregation method, flatten 
    # it and use the group feature as the new row index
    df_pitcher.columns = df_pitcher.columns.get_level_values(0)
    df_pitcher.reset_index(inplace=True)
    df_pitcher.set_index('pitcher')

    velo_df = load_sc_pitching_velo(year, dir)
    spin_df = load_sc_pitching_spin(year, dir) 
    mvmt_df = load_sc_pitching_mvmt(year, dir)

    velo_df.set_index('player_id') 
    spin_df.set_index('player_id') 
    mvmt_df.set_index('pitcher_id') 

    df = df_pitcher.join(velo_df, how='outer', rsuffix='_drop')
    df = df.join(spin_df, how='outer', rsuffix='_drop')
    df = df.join(mvmt_df,  how='outer', rsuffix='_drop')

    df.drop(['player_id_drop'], axis=1, inplace=True)

    return df

def load_sc_batting(year, dir): 
    """
    Load and clean batting data
    """
    sc_df = load_sc_base(year, dir)

    df_batter = sc_df.groupby('batter').agg({
        'pitch_type_CH' : ['sum'], 
        'pitch_type_CS' : ['sum'], 
        'pitch_type_CU' : ['sum'], 
        'pitch_type_EP' : ['sum'],
        'pitch_type_FA' : ['sum'], 
        'pitch_type_FC' : ['sum'], 
        'pitch_type_FF' : ['sum'], 
        'pitch_type_FO' : ['sum'], 
        'pitch_type_FS' : ['sum'], 
        'pitch_type_KC' : ['sum'], 
        'pitch_type_KN' : ['sum'], 
        'pitch_type_PO' : ['sum'], 
        'pitch_type_SC' : ['sum'], 
        'pitch_type_SI' : ['sum'], 
        'pitch_type_SL' : ['sum'], 
        'pitch_type_ST' : ['sum'], 
        'pitch_type_SV' : ['sum'], 
        #'game_date', we don't have temporal data for other statcast metrics, have to retreat to season-level data :(
        'release_speed' : ['mean'], 
        'release_pos_x' : ['mean'], 
        'release_pos_z' : ['mean'], 
        #'batter', # group feature
        #'pitcher', # interesting but would require a one-hot encoding for all pitchers and dealing w/ associated sparseness
        'events_catcher_interf' : ['sum'], 
        'events_double' : ['sum'], 
        'events_double_play' : ['sum'], 
        'events_field_error' : ['sum'], 
        'events_field_out' : ['sum'], 
        'events_fielders_choice' : ['sum'], 
        'events_fielders_choice_out' : ['sum'], 
        'events_force_out' : ['sum'], 
        'events_grounded_into_double_play' : ['sum'], 
        'events_hit_by_pitch' : ['sum'], 
        'events_home_run' : ['sum'], 
        'events_sac_bunt' : ['sum'], 
        'events_sac_fly' : ['sum'], 
        'events_sac_fly_double_play' : ['sum'], 
        'events_single' : ['sum'], 
        'events_strikeout' : ['sum'], 
        'events_strikeout_double_play' : ['sum'], 
        'events_triple' : ['sum'], 
        'events_triple_play' : ['sum'], 
        'events_truncated_pa' : ['sum'], 
        'events_walk' : ['sum'], 
        #'events_None', 
        'description_ball' : ['sum'], 
        'description_blocked_ball' : ['sum'], 
        'description_bunt_foul_tip' : ['sum'], 
        'description_called_strike' : ['sum'], 
        'description_foul' : ['sum'], 
        'description_foul_bunt' : ['sum'], 
        'description_foul_tip' : ['sum'], 
        'description_hit_by_pitch' : ['sum'], 
        'description_hit_into_play' : ['sum'], 
        'description_missed_bunt' : ['sum'], 
        'description_pitchout' : ['sum'], 
        'description_swinging_strike' : ['sum'], 
        'description_swinging_strike_blocked' : ['sum'], 
        #'zone'
        })

    # same trick as for pitching aggregation -- flatten multi-index
    df_batter.columns = df_batter.columns.get_level_values(0)
    df_batter.reset_index(inplace=True)
    df_batter.set_index('batter')

    return df_batter

def load_sc_fielding_cp(year, dir): 
    """
    Load and clean catch probability fielding data
    """

    # We have outs above average, but that is a post-hoc analysis, so leave out (leakage
    # of target label as it requires relative ranking among league)
    file = dir + f"sc_catch_prob_{year}.parquet"
    cdf = None
    try: 
        cdf = pd.read_parquet(file) 

    except FileNotFoundError: 
            
        df = pb.statcast_outfield_catch_prob(year, min_opp=1)

        drop_columns = [
            'last_name, first_name', 
            'year'
            ]

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns)

        cdf.to_parquet(file)

    return cdf

def load_sc_fielding_jump(year, dir): 
    """
    Load and outfielder jump data
    """

    file = dir + f"sc_jump_{year}.parquet"
    cdf = None
    try: 
        cdf = pd.read_parquet(file) 

    except FileNotFoundError: 
            
        df = pb.statcast_outfielder_jump(year, min_att=1)

        drop_columns = [
            'last_name, first_name',
            'outs_avove_average',
            'year'
            ]

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns)
        
        cdf.rename(columns={"resp_fielder_id": "player_id"}, inplace=True)

        cdf.to_parquet(file)

    return cdf

def load_sc_catcher_throwing(year, dir): 
    """
    Load and clean catcher throw data, note this includes
    pop time metrics - no need for separate poptime df 
    """

    file = dir + f"catcher_throwing_{year}.csv"

    cdf = None
    try: 
        df = pd.read_csv(file) 

        onehot_columns = [ 
        ]
        drop_columns = [
            'last_name, first_name'
            ]

        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns, 
            onehot=onehot_columns)

    except FileNotFoundError as e: 
        raise e

    return cdf

def load_sc_catcher_framing(year, dir): 
    """
    Load and clean catcher data
    """

    file = dir + f"sc_framing_{year}.parquet"
    cdf = None
    try: 
        cdf = pd.read_parquet(file) 

    except FileNotFoundError: 
            
        df = pb.statcast_catcher_framing(year, min_called_p=1)

        drop_columns = [
            'player_name', 
            ]

        #TODO: work through columns that need to be normalized and populate above lists
        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns)
        
        cdf['player_id'] = cdf['player_id'].astype('int32')

        cdf.to_parquet(file)

    return cdf

def load_sc_fielding(year, dir): 
    """
    Load all relevant statcast fielding data 
    """

    cp_df = load_sc_fielding_cp(year, dir)
    jump_df = load_sc_fielding_jump(year, dir)
    throwing_df = load_sc_catcher_throwing(year, dir)
    framing_df = load_sc_catcher_framing(year, dir)

    cp_df.set_index('player_id')
    jump_df.set_index('player_id')
    throwing_df.set_index('player_id')
    framing_df.set_index('player_id')

    df = cp_df.join(jump_df, how='outer', rsuffix="_drop")
    df = df.join(throwing_df, how='outer', rsuffix="_drop")
    df = df.join(framing_df, how='outer', rsuffix="_drop")
    df.drop(['player_id_drop'], axis=1, inplace=True)

    # Fill in holes created by above joins
    df.fillna(0)

    return df

def load_sc_running(year, dir): 
    """
    Load and clean baserunning data
    """
    
    file = dir + f"sc_running_{year}.parquet"
    cdf = None
    try: 
        cdf = pd.read_parquet(file) 

    except FileNotFoundError: 
            
        df = pb.statcast_sprint_speed(year, min_opp=1)

        drop_columns = [
            'last_name, first_name',
            'team_id', 
            'team' 
            ]
        onehot_columns = [ 
            'position'
        ]

        #TODO: work through columns that need to be normalized and populate above lists
        cdf = canonicalize_data(
            df, 
            drop_pct=0.01, 
            drop=drop_columns, 
            onehot=onehot_columns   )
        
        cdf.to_parquet(file)

    return cdf

def load_standard(year, dir='data/'):
    """
    Retrieve the conventional/non-statcast metrics for a given year
    """
    
    std_batting = load_std_batting(year, dir) 
    std_pitching = load_std_pitching(year, dir)

    std_batting.set_index('IDfg')
    std_pitching.set_index('IDfg')

    df = std_batting.join(std_pitching, how='outer', rsuffix="_drop")
    df['player_id'] = df['IDfg'].apply(lambda x: idfg_to_mlb(x))
    df.set_index('player_id')

    df.drop(['IDfg_drop', 'IDfg'], axis=1)

    return df

def load_statcast(year, dir='data/'): 
    """
    Load the sc data
    """

    sc_pitch = load_sc_pitching(year, dir)
    sc_batting = load_sc_batting(year, dir)
    sc_fielding = load_sc_fielding(year, dir)
    sc_running = load_sc_running(year, dir)

    sc_pitch.set_index('pitcher')
    sc_batting.set_index('batter')
    sc_fielding.set_index('player_id')
    sc_running.set_index('player_id')

    df = sc_pitch.join(sc_batting, how='outer', rsuffix='_drop')
    df = df.join(sc_fielding, how='outer', rsuffix='_drop')
    df = df.join(sc_running, how='outer', rsuffix='_drop')    
    df.drop(['pitcher', 'batter', 'player_id_drop'], axis=1, inplace=True)
    
    # Fill join holes... 
    df.fillna(0, inplace=True)

    return df
    
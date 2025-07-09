# my_full_etf_analysis.py

import pandas as pd
import numpy as np
import requests
import datetime
from datetime import date
import os
import io
import warnings
import gspread
from google.oauth2 import service_account
from arch import arch_model
from arch.__future__ import reindexing
from scipy.stats import t, chi2, gennorm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
import plotly.io as pio
import traceback

warnings.filterwarnings("ignore")

# Configurazione Plotly per ambienti non Colab (puoi rimuoverlo se usi solo Colab)
try:
    pio.renderers.default = "colab"
except ValueError:
    pio.renderers.default = "notebook" # O "json" per output non interattivi

# --- Variabili globali per API Keys (Verranno passate come argomenti o lette da env nei test/CI) ---
FMP_API_KEY = os.getenv('FMP_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY')
TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
EOD_API_KEY = os.getenv('EOD_API_KEY')

# --- Configurazione Google Sheets (per l'uso reale, non per i test con file locali) ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
# Queste variabili saranno impostate nello script di test o nel CI/CD per l'ambiente reale
SERVICE_ACCOUNT_FILE = None # Verrà sovrascritto
SHEET_ID = None # Verrà sovrascritto
WORKSHEET_NAME = None # Verrà sovrascritto

# --- Funzioni di Caricamento Dati ---
def _load_data_from_google_sheets(sheet_id, worksheet_name, service_account_file_path=None):
    """
    Carica i dati da Google Sheets, con miglioramenti per la robustezza:
    - Gestione dei decimali con virgola.
    - Riconoscimento flessibile della colonna data.
    - Conversione robusta delle colonne numeriche, inclusa 'Chiusura' come 'Close'.
    - Validazione della presenza della colonna 'Close'.
    """
    print(f"Caricamento dati da Google Sheet ID: '{sheet_id}', foglio: '{worksheet_name}'...")
    creds_path = service_account_file_path
    if creds_path is None or not os.path.exists(creds_path):
        print("❌ Il file delle credenziali non è stato caricato o non esiste.")
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)

        list_of_lists = worksheet.get_all_values()

        if not list_of_lists or len(list_of_lists) < 2:
            print(f"❌ Nessun dato o solo header trovato nel foglio '{worksheet_name}'.")
            return None

        headers = [str(h).strip().replace("[","").replace("]","") for h in list_of_lists[0]]
        df = pd.DataFrame(list_of_lists[1:], columns=headers)

        if df.empty:
            print(f"❌ DataFrame vuoto dopo la creazione da list_of_lists per '{worksheet_name}'.")
            return None

        # Riconoscimento flessibile della colonna data/ora
        date_col_candidates = ['Date', 'date', 'Data', 'data', 'Time', 'Timestamp', 'Giorno', 'giorno', 'datetime']
        date_col_found = None
        for dc_candidate in date_col_candidates:
            for actual_col_name in df.columns:
                if actual_col_name.lower() == dc_candidate.lower():
                    date_col_found = actual_col_name
                    break
            if date_col_found: break

        if not date_col_found:
            print(f"❌ Nessuna colonna data/ora riconosciuta. Colonne: {df.columns.tolist()}")
            return None

        try:
            df[date_col_found] = df[date_col_found].astype(str)
            # Tenta più formati data, l'ordine è importante. Usa errors='coerce' per gestire formati non validi
            # Per il formato "GG/MM/YYYY HH.MM.SS" con i punti nell'orario
            df['datetime_converted_temp'] = pd.to_datetime(df[date_col_found], format='%d/%m/%Y %H.%M.%S', errors='coerce')
            
            # Se la conversione sopra ha prodotto troppi NaT, prova altri formati comuni
            if df['datetime_converted_temp'].isna().sum() == len(df):
                df['datetime_converted_temp'] = pd.to_datetime(df[date_col_found], errors='coerce') # Fallback generico

            na_count = df['datetime_converted_temp'].isna().sum()
            if na_count == len(df) and len(df) > 0:
                print(f"❌ TUTTE le date non convertite per '{date_col_found}'. Formato originale esempio: {df[date_col_found].iloc[0] if not df[date_col_found].empty else 'N/A'}")
                return None

            df.set_index('datetime_converted_temp', inplace=True)
            df = df.loc[pd.notna(df.index)]
            if df.empty:
                print(f"❌ DataFrame vuoto dopo conversione data e rimozione NaT.")
                return None
            df.sort_index(inplace=True)
        except Exception as e_date:
            print(f"❌ Errore conversione/impostazione data '{date_col_found}': {e_date}")
            return None

        # Conversione colonne numeriche e gestione 'Close'/'Chiusura'
        cols_to_convert_to_numeric = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume', 'Chiusura': 'Close'}
        final_cols_to_keep = []

        for original_name, target_name in cols_to_convert_to_numeric.items():
            if original_name in df.columns:
                try:
                    df[target_name] = df[original_name].astype(str).str.replace(',', '.', regex=False)
                    # Converte a numerico, imponendo valori nulli se la conversione fallisce
                    df[target_name] = pd.to_numeric(df[target_name], errors='coerce')
                    if target_name not in final_cols_to_keep:
                        final_cols_to_keep.append(target_name)
                except Exception as e_num:
                    print(f"AVVISO: Errore conversione colonna '{original_name}' a numerico: {e_num}. Ignorata.")
            
        # Assicurati che 'Close' sia presente e non tutta NaN
        if 'Close' not in final_cols_to_keep:
            print(f"❌ Colonna 'Close' (o 'Chiusura') non trovata o non convertita correttamente. Colonne disponibili: {df.columns.tolist()}")
            return None
        
        # Filtra solo le colonne desiderate e rimuovi righe con NaN in 'Close'
        df_final = df[final_cols_to_keep].dropna(subset=['Close'])

        if df_final.empty:
            print(f"❌ DataFrame vuoto dopo dropna su 'Close'.")
            return None

        print(f"✅ Dati caricati e processati da '{worksheet_name}'. Colonne: {df_final.columns.tolist()}")
        return df_final.copy()

    except Exception as e:
        print(f"❌ Errore generale in _load_data_from_google_sheets ('{worksheet_name}'): {e}")
        traceback.print_exc()
        return None

def _load_data_from_fmp(symbol, api_key, start_date, end_date):
    print(f"Caricamento dati da FMP per {symbol}...")
    if not api_key: print("API Key FMP mancante."); return None
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'historical' in data and data['historical']:
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in required_cols if col in df.columns]]
            print(f"Dati caricati da FMP per {symbol}.")
            return df
        else: print(f"Nessun dato 'historical' da FMP per {symbol}. Risposta: {data}"); return None
    except requests.exceptions.ReadTimeout:
        print(f"Timeout durante la richiesta a FMP per {symbol}.")
        return None
    except Exception as e: print(f"Errore FMP per {symbol}: {e}"); return None

def _load_data_from_alphavantage(symbol, api_key, start_date, end_date):
    print(f"Caricamento dati da Alpha Vantage per {symbol}...")
    if not api_key: print("API Key Alpha Vantage mancante."); return None
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. adjusted close': 'Adjusted_Close', '6. volume': 'Volume'})
            df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if 'Adjusted_Close' in df.columns and 'Close' not in df.columns:
                df.rename(columns={'Adjusted_Close': 'Close'}, inplace=True)
            df = df[[col for col in required_cols if col in df.columns]]
            print(f"Dati caricati da Alpha Vantage per {symbol}.")
            return df
        elif "Error Message" in data: print(f"Errore Alpha Vantage per {symbol}: {data['Error Message']}"); return None
        elif "Information" in data: print(f"Informazione da Alpha Vantage (potrebbe essere un limite API): {data['Information']}"); return None
        else: print(f"Formato dati Alpha Vantage inatteso per {symbol}. Risposta: {data}"); return None
    except requests.exceptions.ReadTimeout:
        print(f"Timeout durante la richiesta a Alpha Vantage per {symbol}.")
        return None
    except Exception as e: print(f"Errore Alpha Vantage per {symbol}: {e}"); return None

def _load_data_from_twelvedata(symbol, api_key, start_date, end_date):
    print(f"Caricamento dati da Twelve Data per {symbol}...")
    if not api_key: print("API Key Twelve Data mancante."); return None
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&start_date={start_date}&end_date={end_date}&apikey={api_key}&outputsize=5000"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "values" in data and data["values"]:
            df = pd.DataFrame(data["values"])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[[col for col in numeric_cols if col in df.columns]]
            print(f"Dati caricati da Twelve Data per {symbol}.")
            return df
        elif data.get("status") == "error" or data.get("code", 200) != 200 :
            print(f"Errore da Twelve Data per {symbol} ({data.get('code')}): {data.get('message', 'Errore sconosciuto')}")
            return None
        else: print(f"Formato dati Twelve Data inatteso o vuoto per {symbol}. Risposta: {data}"); return None
    except requests.exceptions.ReadTimeout:
        print(f"Timeout durante la richiesta a Twelve Data per {symbol}.")
        return None
    except Exception as e: print(f"Errore Twelve Data per {symbol}: {e}"); return None

def _load_data_from_tiingo(symbol, api_key, start_date, end_date):
    print(f"Caricamento dati da Tiingo per {symbol}...")
    if not api_key: print("API Key Tiingo mancante."); return None
    headers = {'Authorization': f'Token {api_key}'}
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}&format=json"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in required_cols if col in df.columns]]
            print(f"Dati caricati da Tiingo per {symbol}.")
            return df
        else: print(f"Nessun dato da Tiingo per {symbol}. Risposta: {data}"); return None
    except requests.exceptions.ReadTimeout:
        print(f"Timeout durante la richiesta a Tiingo per {symbol}.")
        return None
    except Exception as e: print(f"Errore Tiingo per {symbol}: {e}"); return None

def _load_data_from_eod(symbol, api_key, start_date, end_date):
    print(f"Caricamento dati da EOD Historical Data per {symbol}...")
    if not api_key: print("API Key EOD mancante."); return None
    url = f"https://eodhistoricaldata.com/api/eod/{symbol}?api_token={api_key}&fmt=json&from={start_date}&to={end_date}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in required_cols if col in df.columns]]
            print(f"Dati caricati da EOD Historical Data per {symbol}.")
            return df
        else: print(f"Nessun dato da EOD Historical Data per {symbol}. Risposta: {data}"); return None
    except requests.exceptions.ReadTimeout:
        print(f"Timeout durante la richiesta a EOD Historical Data per {symbol}.")
        return None
    except Exception as e: print(f"Errore EOD Historical Data per {symbol}: {e}"); return None

def load_historical_data(source_type, identifier, start_date, end_date, api_key_dict=None, google_sheets_config=None):
    """Funzione unificata per caricare dati storici da diverse fonti."""
    print(f"Tentativo di caricare dati da {source_type} per {identifier}...")
    api_key_dict = api_key_dict or {}

    if source_type == 'Google Sheets' and google_sheets_config:
        return _load_data_from_google_sheets(
            sheet_id=google_sheets_config.get('sheet_id'),
            worksheet_name=google_sheets_config.get('worksheet_name'),
            service_account_file_path=google_sheets_config.get('service_account_file_path')
        )
    elif source_type == 'FMP':
        return _load_data_from_fmp(identifier, api_key_dict.get('FMP_API_KEY'), start_date, end_date)
    elif source_type == 'AlphaVantage':
        return _load_data_from_alphavantage(identifier, api_key_dict.get('ALPHA_VANTAGE_API_KEY'), start_date, end_date)
    elif source_type == 'TwelveData':
        return _load_data_from_twelvedata(identifier, api_key_dict.get('TWELVE_DATA_API_KEY'), start_date, end_date)
    elif source_type == 'Tiingo':
        return _load_data_from_tiingo(identifier, api_key_dict.get('TIINGO_API_KEY'), start_date, end_date)
    elif source_type == 'EOD':
        return _load_data_from_eod(identifier, api_key_dict.get('EOD_API_KEY'), start_date, end_date)
    else:
        print(f"❌ Sorgente dati '{source_type}' non supportata o configurazione incompleta.")
        return None

# --- Funzioni di Analisi Statistica e Simulazione ---

def check_normality(data, alpha=0.05):
    """
    Verifica la normalità dei residui usando il test di Jarque-Bera.
    """
    if len(data) < 2:
        return False, "Dati insufficienti per il test di normalità."
    jb_test = stats.jarque_bera(data)
    is_normal = jb_test[1] > alpha
    return is_normal, f"Jarque-Bera p-value: {jb_test[1]:.4f}"

def check_autocorrelation(data, lags=10, alpha=0.05):
    """
    Verifica l'autocorrelazione dei residui usando il test di Ljung-Box.
    """
    if len(data) < lags + 1:
        return False, "Dati insufficienti per il test di autocorrelazione con i lag specificati."
    lb_test = acorr_ljungbox(data, lags=[lags], return_df=True)
    is_autocorrelated = lb_test['lb_pvalue'].iloc[0] < alpha
    return not is_autocorrelated, f"Ljung-Box p-value (lag {lags}): {lb_test['lb_pvalue'].iloc[0]:.4f}"

def estimate_garch_parameters(returns):
    """
    Stima i parametri di un modello GARCH(1,1) sui rendimenti.
    """
    if returns is None or returns.empty:
        print("❌ Rendimenti non disponibili per la stima GARCH.")
        return None, None
    if len(returns) < 5: # Minimo di punti per una stima sensata, anche se teoricamente meno
        print(f"❌ Dati insufficienti ({len(returns)} punti) per la stima GARCH. Richiesti almeno 5.")
        return None, None
    
    # Rimuovi eventuali NaN o infiniti dai rendimenti
    returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns_clean.empty:
        print("❌ Rendimenti puliti vuoti dopo la rimozione di NaN/Inf per la stima GARCH.")
        return None, None

    try:
        model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='StudentsT')
        res = model.fit(disp='off')
        # print(res.summary()) # Commentato per non inondare l'output durante i test
        return res, True
    except Exception as e:
        print(f"❌ Errore durante la stima del modello GARCH: {e}")
        return None, False

def simulate_garch_prices(garch_result, last_price, num_simulations=1000, days_to_forecast=252):
    """
    Simula i prezzi futuri usando il modello GARCH stimato.
    """
    if garch_result is None:
        print("❌ Risultato GARCH nullo, impossibile simulare.")
        return None

    try:
        # Usa forecast per ottenere le volalitità condizionate
        # horizon=days_to_forecast per ottenere previsioni giornaliere per tutti i giorni
        # method='simulation' e simulations=num_simulations per generare scenari
        forecasts = garch_result.forecast(horizon=days_to_forecast, method='simulation', simulations=num_simulations,
                                         reindex=False) # reindex=False per evitare warning
        
        # Le simulazioni di rendimento sono in forecasts.simulations.values
        # Ha forma (days_to_forecast, num_simulations, 1) -> (giorni, simulazioni, 1)
        simulated_returns = forecasts.simulations.values[0, :, :] # Prendi i rendimenti dal primo (e unico) set di orizzonte
        simulated_returns = pd.DataFrame(simulated_returns) # Converti a DataFrame per facilità

        # Inizializza i prezzi simulati
        simulated_prices = np.zeros((days_to_forecast, num_simulations))
        simulated_prices[0, :] = last_price * np.exp(simulated_returns.iloc[0])

        for t in range(1, days_to_forecast):
            simulated_prices[t, :] = simulated_prices[t-1, :] * np.exp(simulated_returns.iloc[t])

        return simulated_prices
    except Exception as e:
        print(f"❌ Errore durante la simulazione dei prezzi GARCH: {e}")
        traceback.print_exc()
        return None

def calculate_metrics(df_result):
    """
    Calcola il rendimento medio e la volatilità annualizzata dal DataFrame dei risultati.
    """
    if df_result is None or df_result.empty or 'Rendimento' not in df_result.columns:
        print("❌ DataFrame risultati non valido o colonna 'Rendimento' mancante per il calcolo delle metriche.")
        return None, None
    
    # Rimuovi eventuali NaN o infiniti prima del calcolo
    returns_clean = df_result['Rendimento'].replace([np.inf, -np.inf], np.nan).dropna()
    if returns_clean.empty:
        print("❌ Nessun rendimento valido dopo la pulizia per il calcolo delle metriche.")
        return None, None

    daily_return = returns_clean.mean()
    daily_volatility = returns_clean.std()

    # Annualizzazione (considerando 252 giorni di trading in un anno)
    annualized_return = daily_return * 252
    annualized_volatility = daily_volatility * np.sqrt(252)

    return annualized_return, annualized_volatility

def plot_prezzi_simulati_plotly(simulated_prices, titolo="Andamento Prezzi Simulati", start_date=None, historical_prices=None):
    """Genera un fan chart interattivo dei prezzi simulati usando Plotly."""
    if simulated_prices is None or simulated_prices.size == 0:
        print("Nessun prezzo simulato da plottare.")
        return None

    if start_date is None:
        start_date = pd.Timestamp.today()

    giorni_forecast = simulated_prices.shape[0]
    forecast_dates_axis = pd.bdate_range(start=start_date, periods=giorni_forecast, freq='B')

    quantili_plot = [5, 25, 50, 75, 95]
    price_percentiles = np.percentile(simulated_prices, quantili_plot, axis=1)

    fig = go.Figure()

    p_map = {q: price_percentiles[i] for i, q in enumerate(quantili_plot)}

    # Fascia esterna (5-95)
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_dates_axis, forecast_dates_axis[::-1]]),
        y=np.concatenate([p_map[95], p_map[5][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(width=0),
        name='5-95 Percentile',
        hoverinfo='skip'
    ))
    # Fascia interna (25-75, IQR)
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_dates_axis, forecast_dates_axis[::-1]]),\
        y=np.concatenate([p_map[75], p_map[25][::-1]]),\
        fill='toself',\
        fillcolor='rgba(0,100,200,0.4)',\
        line=dict(width=0),\
        name='25-75 Percentile (IQR)',\
        hoverinfo='skip'\
    ))\

    # Linea della Mediana (50° percentile)
    fig.add_trace(go.Scatter(
        x=forecast_dates_axis,
        y=p_map[50],
        mode='lines',
        name='Mediana (50%)',
        line=dict(color='navy', width=2)
    ))

    # Prezzi storici
    if historical_prices is not None and not historical_prices.empty:
        fig.add_trace(go.Scatter(
            x=historical_prices.index,
            y=historical_prices,
            mode='lines',
            name='Prezzi Storici',
            line=dict(color='black', width=2)
        ))

    fig.update_layout(
        title_text=titolo,
        xaxis_title="Data",
        yaxis_title="Prezzo Stimato",
        template="plotly_white",
        height=500,
        showlegend=True,
        hovermode="x unified"
    )

    return fig

# Funzione Principale di Analisi ETF
def analizza_etf_veloce(df_input_raw, days_to_forecast=252, num_simulations=1000, 
                         api_key_dict=None, google_sheets_config=None, source_type=None, identifier=None):
    """
    Funzione principale per l'analisi rapida di un ETF o dati di prezzo.
    Accetta un DataFrame grezzo (come da Google Sheets o CSV) o carica da API.
    """
    df = df_input_raw
    # Se df_input_raw è None, prova a caricare dai parametri API/Sheets
    if df is None and source_type and identifier:
        print(f"Tentativo di caricare dati esterni per {identifier} da {source_type}...")
        # Usa date fisse per l'esempio, dovresti parametrizzarle o caricarle in base alla tua logica
        _start_date = (date.today() - datetime.timedelta(days=365*2)).strftime('%Y-%m-%d')
        _end_date = date.today().strftime('%Y-%m-%d')
        df = load_historical_data(source_type, identifier, _start_date, _end_date, api_key_dict, google_sheets_config)
    
    if df is None or df.empty:
        print("❌ Impossibile ottenere dati validi per l'analisi.")
        return None, None # Restituisce None per entrambi i valori in caso di fallimento

    # Assicurati che la colonna 'Close' sia numerica e pulita
    if 'Close' not in df.columns:
        print("❌ Colonna 'Close' mancante nel DataFrame.")
        return None, None
    
    # Tenta una conversione robusta in caso di errori pregressi o tipi misti
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
    except Exception as e:
        print(f"❌ Errore nella conversione robusta della colonna 'Close': {e}")
        return None, None

    if df.empty:
        print("❌ DataFrame vuoto dopo la pulizia della colonna 'Close'.")
        return None, None

    # Calcolo dei rendimenti logaritmici
    if len(df['Close']) < 2:
        print("❌ Dati insufficienti per calcolare i rendimenti (almeno 2 punti necessari).")
        return None, None
    df['Rendimento'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()

    if df['Rendimento'].empty:
        print("❌ Rendimenti non calcolabili o vuoti dopo la dropna.")
        return None, None
    
    # Pulizia rendimenti da valori infiniti/estremamente grandi
    df['Rendimento'] = df['Rendimento'].replace([np.inf, -np.inf], np.nan).dropna()
    if df['Rendimento'].empty:
        print("❌ Rendimenti vuoti dopo la pulizia di infiniti.")
        return None, None


    # Stima del modello GARCH
    garch_res, garch_success = estimate_garch_parameters(df['Rendimento'])

    if not garch_success:
        print("❌ Stima GARCH fallita. Impossibile procedere con la simulazione.")
        return None, None

    # Simulazione prezzi futuri
    last_price = df['Close'].iloc[-1]
    simulated_prices = simulate_garch_prices(garch_res, last_price, num_simulations, days_to_forecast)

    if simulated_prices is None:
        print("❌ Simulazione prezzi fallita.")
        return None, None

    # Calcolo metriche
    annualized_return, annualized_volatility = calculate_metrics(df) # Usa il df originale con la colonna 'Rendimento'

    # Plot (opzionale, per un ambiente reale)
    # fig = plot_prezzi_simulati_plotly(simulated_prices, historical_prices=df['Close'])
    # if fig: fig.show()

    return annualized_return, annualized_volatility

# Blocchi di codice interattivi del notebook (li ho lasciati commentati per coerenza)
# Questi blocchi verrebbero eseguiti nel notebook, ma non nel contesto di un test
# o di un'esecuzione automatica, dove i parametri sono fissi o passati via CLI/ambiente.

# Esempio di utilizzo (per debugging o esecuzione manuale)
if __name__ == "__main__":
    print("Esempio di esecuzione manuale di analizza_etf_veloce (senza simulazione API/Sheets).")
    # Crea un DataFrame di esempio per il test
    sample_data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10']),
        'Close': [100.0, 101.5, 100.8, 102.1, 103.5, 102.9, 104.0, 105.1, 104.5, 106.0]
    }
    df_sample = pd.DataFrame(sample_data).set_index('Date')

    # Chiamata alla funzione principale
    ret, vol = analizza_etf_veloce(df_sample)

    if ret is not None and vol is not None:
        print(f"Rendimento Annualizzato Stimato: {ret:.4f}")
        print(f"Volatilità Annualizzata Stimata: {vol:.4f}")
    else:
        print("Analisi fallita per il DataFrame di esempio.")

    print("\nEsempio di caricamento dati simulato da Google Sheets:")
    # Per questo test, useremo un DataFrame simulato invece di un vero foglio Google
    # In un ambiente di test reale, si mockerebbe la chiamata a gspread
    simulated_gs_data = {
        'Data': ['01/01/2023 10.00.00', '02/01/2023 10.00.00', '03/01/2023 10.00.00', '04/01/2023 10.00.00', '05/01/2023 10.00.00'],
        'Chiusura': ['100,5', '101,2', '100,9', '102,5', '103,1'],
        'Volume': ['1000', '1200', '1100', '1500', '1300']
    }
    simulated_df_gs = pd.DataFrame(simulated_gs_data)

    # df_from_gs_sim = _load_data_from_google_sheets(
    #     sheet_id="dummy_id", worksheet_name="dummy_name", 
    #     service_account_file_path="dummy_creds.json"
    # )
    # print(df_from_gs_sim.head() if df_from_gs_sim is not None else "Caricamento simulato fallito.")

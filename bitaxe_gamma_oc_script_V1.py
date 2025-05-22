import requests
import time
import csv
import statistics
import sys
import re
import argparse
import logging
import signal
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# Configuration par défaut
# -----------------------

MINER_IP = "192.168.1.165"
RESULTS_CSV = "bitaxe_testing.csv"

FREQ_START = 525
FREQ_END = 875
FREQ_STEP = 5

CV_START = 1150
CV_MAX = 1250
CV_STEP = 10

SETTLE_TIME = 180
MEASURE_DURATION = 180
MEASURE_INTERVAL = 1

CONFIRM_DURATION = 60
CONFIRM_INTERVAL = 1
CONFIRM_ATTEMPTS = 2

TEMP_LIMIT = 60
HASHRATE_TOLERANCE = 0.90
COEF_VARIATION_THRESHOLD = 0.12

# -----------------------
# Logging
# -----------------------

logging.basicConfig(
    filename='bitaxe_tuning.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def log_info(msg):
    print(msg, flush=True)
    logging.info(msg)

def log_error(msg):
    print(msg, flush=True)
    logging.error(msg)

# -----------------------
# Utilitaires
# -----------------------

def validate_ip(ip):
    pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    return re.match(pattern, ip) is not None and ip != "REPLACE_WITH_YOUR_BITAXE_IP"

def safe_request(method, url, **kwargs):
    try:
        response = requests.request(method, url, timeout=10, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        log_error(f"[ERREUR] {method} {url} échoué : {e}")
        return None

def signal_handler(sig, frame):
    log_info("\n[INFO] Arrêt demandé par l'utilisateur. Sortie propre...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# -----------------------
# Fonctions principales
# -----------------------

def set_miner_settings(freq, cv, patch_url):
    payload = {
        "frequency": freq,
        "coreVoltage": cv,
        "autoFanSpeed": True,
        "flipScreen": True,
        "invertFanPolarity": True,
    }
    log_info(f"[INFO] Patching: freq={freq} MHz, cv={cv} mV...")
    response = safe_request("PATCH", patch_url, json=payload)
    if response:
        log_info(f"[INFO] PATCH success: freq={freq}, cv={cv}")

def get_miner_stats(stats_url):
    response = safe_request("GET", stats_url)
    if response:
        data = response.json()
        return float(data.get("hashRate", 0)), float(data.get("temp", 0))
    return None, None

def measure_hashrate_stats(duration, interval, stats_url):
    steps = duration // interval
    samples = []
    last_temp = 0

    log_info(f"[INFO] Measuring for {duration}s (interval: {interval}s)...")
    for i in tqdm(range(steps), desc="Mesure", ncols=70):
        time.sleep(interval)
        hr, temp = get_miner_stats(stats_url)
        hr = hr if hr is not None else 0
        temp = temp if temp is not None else last_temp

        samples.append(hr)
        last_temp = temp

        if (i + 1) % 30 == 0 or i == steps - 1:
            log_info(f"  [DEBUG] Sample {i+1}/{steps}: hr={hr:.2f}, temp={temp:.2f}")

        if temp >= TEMP_LIMIT:
            log_info(f"[WARNING] Température atteinte {temp}°C. Arrêt de la mesure...")
            return None, None, temp, True

    avg = sum(samples) / len(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0
    return avg, std, last_temp, False

def confirm_drop(threshold, attempts, freq, cv, best_hashrate, stats_url):
    for attempt in range(1, attempts + 1):
        log_info(f"[INFO] Confirmation de baisse ({attempt}/{attempts}) pour freq={freq} MHz, cv={cv} mV...")
        avg, std, temp, aborted = measure_hashrate_stats(CONFIRM_DURATION, CONFIRM_INTERVAL, stats_url)
        if aborted:
            return None, None, True

        coef = (std / avg) if avg > 0 else 0
        log_info(f"[INFO] Confirm #{attempt}: avg={avg:.2f}, stdev={std:.2f}, coef={coef:.2f}")

        if coef > COEF_VARIATION_THRESHOLD or avg >= best_hashrate * threshold:
            return avg, coef, False

    return avg, coef, False

# -----------------------
# Génération du graphique
# -----------------------

def export_graph(results, output_png):
    freqs = [r["frequency"] for r in results]
    hashrates = [r["hashrate"] for r in results]
    temps = [r["temperature"] for r in results]
    cvs = [r["coreVoltage"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, hashrates, marker='o', label="Hashrate (H/s)")
    plt.ylabel("Hashrate (H/s)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs, temps, marker='x', color='r', label="Température (°C)")
    plt.plot(freqs, cvs, marker='s', color='g', label="Tension (mV)")
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Température / Tension")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_png)
    log_info(f"[INFO] Graphique exporté : {output_png}")

# -----------------------
# Main Tuning Logic
# -----------------------

def main(args):
    miner_ip = args.ip
    results_csv = args.csv
    freq_start = args.freq_start
    freq_end = args.freq_end
    freq_step = args.freq_step
    cv_start = args.cv_start
    cv_max = args.cv_max
    cv_step = args.cv_step
    settle_time = args.settle_time
    measure_duration = args.measure_duration
    measure_interval = args.measure_interval
    confirm_duration = args.confirm_duration
    confirm_interval = args.confirm_interval
    confirm_attempts = args.confirm_attempts
    temp_limit = args.temp_limit
    hashrate_tolerance = args.hashrate_tolerance
    coef_variation_threshold = args.coef_variation_threshold
    output_png = args.png

    global TEMP_LIMIT, HASHRATE_TOLERANCE, COEF_VARIATION_THRESHOLD
    TEMP_LIMIT = temp_limit
    HASHRATE_TOLERANCE = hashrate_tolerance
    COEF_VARIATION_THRESHOLD = coef_variation_threshold

    if not validate_ip(miner_ip):
        log_error("[ERREUR] Veuillez configurer une adresse IP valide pour votre Bitaxe.")
        sys.exit(1)

    PATCH_URL = f"http://{miner_ip}/api/system"
    STATS_URL = f"http://{miner_ip}/api/system/info"

    log_info("[INFO] Démarrage du tuning Bitaxe...")

    current_freq = freq_start
    current_cv = cv_start

    set_miner_settings(current_freq, current_cv, PATCH_URL)
    log_info(f"[INFO] Attente de {settle_time}s pour stabilisation...")
    time.sleep(settle_time)

    baseline, std_base, temp, aborted = measure_hashrate_stats(measure_duration, measure_interval, STATS_URL)
    if aborted or baseline is None:
        log_info("[INFO] Abandon : baseline échoué ou température trop élevée.")
        return

    best_hashrate = baseline
    log_info(f"[INFO] Baseline hashrate: {baseline:.2f} H/s")

    results = [{
        "frequency": current_freq,
        "coreVoltage": current_cv,
        "hashrate": baseline,
        "temperature": temp,
        "stdev": std_base
    }]

    for freq in range(freq_start + freq_step, freq_end + 1, freq_step):
        log_info(f"\n[INFO] Test freq={freq} MHz à cv={current_cv} mV...")
        set_miner_settings(freq, current_cv, PATCH_URL)
        log_info(f"[INFO] Attente {settle_time}s...")
        time.sleep(settle_time)

        avg, std, temp, aborted = measure_hashrate_stats(measure_duration, measure_interval, STATS_URL)
        if aborted:
            break

        coef = (std / avg) if avg > 0 else 0
        log_info(f"[INFO] Résultat: avg={avg:.2f}, stdev={std:.2f}, coef={coef:.2f}")

        if coef <= coef_variation_threshold and avg < best_hashrate * hashrate_tolerance:
            log_info("[INFO] Sous-voltage suspecté — confirmation...")
            confirm_avg, confirm_coef, confirm_abort = confirm_drop(
                hashrate_tolerance, confirm_attempts, freq, current_cv, best_hashrate, STATS_URL
            )

            if confirm_abort or confirm_avg is None:
                continue

            if confirm_avg < best_hashrate * hashrate_tolerance:
                log_info("[INFO] Baisse confirmée — augmentation de la tension...")
                while current_cv < cv_max and confirm_avg < best_hashrate * hashrate_tolerance:
                    current_cv += cv_step
                    log_info(f"  [INFO] Augmentation tension à {current_cv} mV")
                    set_miner_settings(freq, current_cv, PATCH_URL)
                    time.sleep(settle_time)

                    confirm_avg, std_temp, temp, aborted_voltage = measure_hashrate_stats(
                        measure_duration, measure_interval, STATS_URL
                    )
                    if aborted_voltage:
                        break

                    confirm_avg, confirm_coef, confirm_abort = confirm_drop(
                        hashrate_tolerance, confirm_attempts, freq, current_cv, best_hashrate, STATS_URL
                    )
                    if confirm_abort or confirm_avg is None:
                        break

        results.append({
            "frequency": freq,
            "coreVoltage": current_cv,
            "hashrate": avg,
            "temperature": temp,
            "stdev": std
        })

        if coef <= coef_variation_threshold and avg > best_hashrate:
            best_hashrate = avg
            log_info(f"[INFO] Nouveau meilleur hashrate: {best_hashrate:.2f}")

        if temp >= temp_limit:
            log_info("[INFO] Arrêt — limite de température atteinte.")
            break

    log_info("[INFO] Écriture des résultats dans le CSV...")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frequency", "coreVoltage", "hashrate", "temperature", "stdev"])
        writer.writeheader()
        writer.writerows(results)

    log_info(f"[INFO] Terminé ! Résultats sauvegardés dans {results_csv}")

    # Génération du graphique
    export_graph(results, output_png)

# -----------------------
# Entry Point
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Bitaxe Gamma OC Script")
    parser.add_argument("--ip", type=str, default=MINER_IP, help="Adresse IP du Bitaxe")
    parser.add_argument("--csv", type=str, default=RESULTS_CSV, help="Fichier de sortie CSV")
    parser.add_argument("--png", type=str, default="bitaxe_tuning_graph.png", help="Fichier de sortie graphique PNG")
    parser.add_argument("--freq_start", type=int, default=FREQ_START, help="Fréquence de départ (MHz)")
    parser.add_argument("--freq_end", type=int, default=FREQ_END, help="Fréquence de fin (MHz)")
    parser.add_argument("--freq_step", type=int, default=FREQ_STEP, help="Pas de fréquence (MHz)")
    parser.add_argument("--cv_start", type=int, default=CV_START, help="Tension de départ (mV)")
    parser.add_argument("--cv_max", type=int, default=CV_MAX, help="Tension max (mV)")
    parser.add_argument("--cv_step", type=int, default=CV_STEP, help="Pas de tension (mV)")
    parser.add_argument("--settle_time", type=int, default=SETTLE_TIME, help="Temps de stabilisation (s)")
    parser.add_argument("--measure_duration", type=int, default=MEASURE_DURATION, help="Durée de mesure (s)")
    parser.add_argument("--measure_interval", type=int, default=MEASURE_INTERVAL, help="Intervalle de mesure (s)")
    parser.add_argument("--confirm_duration", type=int, default=CONFIRM_DURATION, help="Durée de confirmation (s)")
    parser.add_argument("--confirm_interval", type=int, default=CONFIRM_INTERVAL, help="Intervalle de confirmation (s)")
    parser.add_argument("--confirm_attempts", type=int, default=CONFIRM_ATTEMPTS, help="Nombre de confirmations")
    parser.add_argument("--temp_limit", type=int, default=TEMP_LIMIT, help="Limite de température (°C)")
    parser.add_argument("--hashrate_tolerance", type=float, default=HASHRATE_TOLERANCE, help="Tolérance hashrate (0-1)")
    parser.add_argument("--coef_variation_threshold", type=float, default=COEF_VARIATION_THRESHOLD, help="Seuil de variation du coef")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

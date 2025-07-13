import configparser
import os
import numpy as np

def load_constants(filename="../config/constants.dat"):
    # Check for environment variable override
    if 'CONSTANTS_PATH' in os.environ:
        filename = os.environ['CONSTANTS_PATH']

    if not os.path.exists(filename):
        # Try fallback path
        fallback = "./config/constants.dat"
        if os.path.exists(fallback):
            filename = fallback
        else:
            raise FileNotFoundError(f"Could not find {filename} or {fallback}. Check file existence or set CONSTANTS_PATH environment variable.")

    config = configparser.ConfigParser(
        comment_prefixes=('#', ';'),
        inline_comment_prefixes=('#', ';'),
        empty_lines_in_values=False
    )
    # Custom parser to handle ': Description, Units'
    with open(filename, 'r') as f:
        lines = f.readlines()
    config_lines = []
    for line in lines:
        if line.strip() and not line.strip().startswith(('#', ';', '-------')):
            if ':' in line:
                config_lines.append(line.split(':', 1)[0].strip())
            else:
                config_lines.append(line.strip())
    with open('temp_config.ini', 'w') as f:
        f.write('\n'.join(config_lines))
    config.read('temp_config.ini')
    os.remove('temp_config.ini')
    if 'Constants' not in config:
        raise ValueError(f"Could not find [Constants] section in {filename}")

    # Compute G_GAMMA_POINTS based on range and points per decade
    g_gamma_log_range = float(config['Constants']['G_GAMMA_LOG_MAX']) - float(config['Constants']['G_GAMMA_LOG_MIN'])
    g_gamma_points = int(np.ceil(g_gamma_log_range * int(config['Constants']['G_GAMMA_POINTS_PER_DECADE'])))
    print(f"DEBUG: G_GAMMA_POINTS computed as {g_gamma_points} (range={g_gamma_log_range}, points_per_decade={int(config['Constants']['G_GAMMA_POINTS_PER_DECADE'])})")

    # Override DATA_DIR to point to main project data/ folder
    constants = {
        'DATA_DIR': os.path.normpath('../data/'),
        'N_REALIZATIONS': int(config['Constants']['N_REALIZATIONS']),
        'SOLAR_NU_LIMIT': float(config['Constants']['SOLAR_NU_LIMIT']),
        'MAX_N': int(config['Constants']['MAX_N']),
        'G_E_LOG_MIN': float(config['Constants']['G_E_LOG_MIN']),
        'G_E_LOG_MAX': float(config['Constants']['G_E_LOG_MAX']),
        'G_GAMMA_LOG_MIN': float(config['Constants']['G_GAMMA_LOG_MIN']),
        'G_GAMMA_LOG_MAX': float(config['Constants']['G_GAMMA_LOG_MAX']),
        'G_E_POINTS_PER_DECADE': int(config['Constants']['G_E_POINTS_PER_DECADE']),
        'G_GAMMA_POINTS_PER_DECADE': int(config['Constants']['G_GAMMA_POINTS_PER_DECADE']),
        'G_GAMMA_POINTS': g_gamma_points,
        'G_GAMMA_N1': float(config['Constants']['G_GAMMA_N1']),
        'G_GAMMA_G_E_LIMIT': float(config['Constants']['G_GAMMA_G_E_LIMIT']),
        'PHI_P_COEFF': float(config['Constants']['PHI_P_COEFF']),
        'PHI_B_COEFF': float(config['Constants']['PHI_B_COEFF']),
        'PHI_C_COEFF': float(config['Constants']['PHI_C_COEFF'])
    }

    # Validate numeric constants
    if constants['N_REALIZATIONS'] <= 0:
        raise ValueError("N_REALIZATIONS must be positive")
    if constants['MAX_N'] < 2:
        raise ValueError("MAX_N must be at least 2")
    if constants['G_E_POINTS_PER_DECADE'] <= 0:
        raise ValueError("G_E_POINTS_PER_DECADE must be positive")
    if constants['G_GAMMA_POINTS_PER_DECADE'] <= 0:
        raise ValueError("G_GAMMA_POINTS_PER_DECADE must be positive")
    if constants['G_E_LOG_MIN'] >= constants['G_E_LOG_MAX']:
        raise ValueError("G_E_LOG_MIN must be less than G_E_LOG_MAX")
    if constants['G_GAMMA_LOG_MIN'] >= constants['G_GAMMA_LOG_MAX']:
        raise ValueError("G_GAMMA_LOG_MIN must be less than G_GAMMA_LOG_MAX")
    if constants['G_GAMMA_POINTS'] <= 0:
        raise ValueError("G_GAMMA_POINTS must be positive")

    return constants

CONSTANTS = load_constants()
DATA_DIR = CONSTANTS['DATA_DIR']
N_REALIZATIONS = CONSTANTS['N_REALIZATIONS']
SOLAR_NU_LIMIT = CONSTANTS['SOLAR_NU_LIMIT']
MAX_N = CONSTANTS['MAX_N']
G_E_LOG_MIN = CONSTANTS['G_E_LOG_MIN']
G_E_LOG_MAX = CONSTANTS['G_E_LOG_MAX']
G_GAMMA_LOG_MIN = CONSTANTS['G_GAMMA_LOG_MIN']
G_GAMMA_LOG_MAX = CONSTANTS['G_GAMMA_LOG_MAX']
G_E_POINTS_PER_DECADE = CONSTANTS['G_E_POINTS_PER_DECADE']
G_GAMMA_POINTS_PER_DECADE = CONSTANTS['G_GAMMA_POINTS_PER_DECADE']
G_GAMMA_POINTS = CONSTANTS['G_GAMMA_POINTS']
G_GAMMA_N1 = CONSTANTS['G_GAMMA_N1']
G_GAMMA_G_E_LIMIT = CONSTANTS['G_GAMMA_G_E_LIMIT']
PHI_P_COEFF = CONSTANTS['PHI_P_COEFF']
PHI_B_COEFF = CONSTANTS['PHI_B_COEFF']
PHI_C_COEFF = CONSTANTS['PHI_C_COEFF']

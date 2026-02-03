"""
Manual Prediction Validation
For when you manually enter actual results from ESPN/NBA.com
"""

import pandas as pd
from datetime import datetime
import os


def load_predictions(prediction_date=None):
    """Load prediction file"""
    if prediction_date is None:
        prediction_date = datetime.now().strftime('%Y%m%d')

    prediction_file = f"./results/predictions_{prediction_date}.csv"

    if not os.path.exists(prediction_file):
        import glob
        files = glob.glob("./results/predictions_*.csv")
        if files:
            prediction_file = max(files, key=os.path.getctime)
            print(f"Using: {prediction_file}")

    return pd.read_csv(prediction_file)


def create_validation_template(predictions_df, output_file='./results/validation_template.csv'):
    """
    Create template CSV for manual entry
    """
    df = predictions_df.copy()

    # -------------------------------------------------
    # REQUIRED COLUMN CHECK (EXPLICIT & SAFE)
    # -------------------------------------------------
    required_columns = {
        'PLAYER_NAME': 'Player',
        'TEAM_ABBREVIATION': 'Team',
        'PTS_PREDICTED': 'PTS',
        'REB_PREDICTED': 'REB',
        'AST_PREDICTED': 'AST',
    }

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in predictions file: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Rename to standard validation schema
    df = df.rename(columns=required_columns)

    # -------------------------------------------------
    # BUILD TEMPLATE
    # -------------------------------------------------
    template = df[['Player', 'Team', 'PTS', 'REB', 'AST']].copy()

    template['PTS_ACTUAL'] = ''
    template['REB_ACTUAL'] = ''
    template['AST_ACTUAL'] = ''
    template['DID_PLAY'] = 'YES'

    # Final column order
    template = template[
        [
            'Player', 'Team',
            'PTS', 'PTS_ACTUAL',
            'REB', 'REB_ACTUAL',
            'AST', 'AST_ACTUAL',
            'DID_PLAY'
        ]
    ]

    template.to_csv(output_file, index=False)

    print(f"\n✓ Template created: {output_file}")
    print("\nInstructions:")
    print("1. Open this file in Excel/Google Sheets")
    print("2. Fill in PTS_ACTUAL, REB_ACTUAL, AST_ACTUAL columns")
    print("3. Mark DID_PLAY='NO' for players who didn't play")
    print("4. Save the file")
    print("5. Run: python scripts/validate_manual.py validate")

    return template


def validate_from_template(template_file='./results/validation_template.csv'):
    """
    Validate predictions from filled template
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "VALIDATION REPORT")
    print("=" * 70)

    df = pd.read_csv(template_file)

    df = df[df['DID_PLAY'].str.upper() == 'YES'].copy()

    for col in ['PTS_ACTUAL', 'REB_ACTUAL', 'AST_ACTUAL']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['PTS_ACTUAL', 'REB_ACTUAL', 'AST_ACTUAL'])

    if df.empty:
        print("\n❌ No valid data found!")
        return

    print(f"\n✓ Validating {len(df)} players\n")

    # Errors
    df['PTS_ERROR'] = abs(df['PTS'] - df['PTS_ACTUAL'])
    df['REB_ERROR'] = abs(df['REB'] - df['REB_ACTUAL'])
    df['AST_ERROR'] = abs(df['AST'] - df['AST_ACTUAL'])
    df['TOTAL_ERROR'] = df[['PTS_ERROR', 'REB_ERROR', 'AST_ERROR']].sum(axis=1)

    print("📊 OVERALL PERFORMANCE\n")
    print(f"{'Stat':<6} {'MAE':<8} {'Within 3':<12}")
    print("-" * 40)

    for stat in ['PTS', 'REB', 'AST']:
        mae = df[f'{stat}_ERROR'].mean()
        within_3 = (df[f'{stat}_ERROR'] <= 3).mean() * 100
        print(f"{stat:<6} {mae:<8.2f} {within_3:<12.1f}%")

    # Save results
    date = datetime.now().strftime('%Y%m%d')
    out = f"./results/validation_results_{date}.csv"
    df.to_csv(out, index=False)

    print(f"\n✓ Results saved: {out}")
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE!")
    print("=" * 70)


def main():
    import sys

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║              MANUAL PREDICTION VALIDATION                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/validate_manual.py template")
        print("  python scripts/validate_manual.py validate")
        return

    if sys.argv[1].lower() == 'template':
        df = load_predictions()
        create_validation_template(df)
    elif sys.argv[1].lower() == 'validate':
        validate_from_template()
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()

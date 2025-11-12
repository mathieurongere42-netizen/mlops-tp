# src/monitor.py (version Windows compatible)
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Important pour Windows sans interface graphique
import matplotlib.pyplot as plt
import seaborn as sns
import os

def monitor_data_quality():
    """Surveille la qualité des données"""
    print("📊 Monitoring de la qualité des données...")
    
    os.makedirs('reports', exist_ok=True)
    
    try:
        train_df = pd.read_csv(os.path.join('data', 'train.csv'))
        test_df = pd.read_csv(os.path.join('data', 'test.csv'))
        
        # Rapport simple sans visualisations complexes
        quality_report = {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_negative': len(train_df[train_df['sentiment'] == 0]),
            'train_positive': len(train_df[train_df['sentiment'] == 1]),
            'test_negative': len(test_df[test_df['sentiment'] == 0]),
            'test_positive': len(test_df[test_df['sentiment'] == 1]),
        }
        
        quality_df = pd.DataFrame([quality_report])
        quality_df.to_csv('reports/data_quality_report.csv', index=False)
        
        print("✅ Monitoring terminé.")
        for key, value in quality_report.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Erreur lors du monitoring: {e}")

if __name__ == "__main__":
    monitor_data_quality()
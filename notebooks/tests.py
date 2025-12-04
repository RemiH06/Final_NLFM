# test_banxico_api.py
import sys
sys.path.append('src')

from data_acquisition import GetBanxicoToken, FetchSeriesData

print("="*60)
print("PRUEBA DE CONEXIÓN CON BANXICO API")
print("="*60)

# 1. Leer token
try:
    token = GetBanxicoToken('.secrets')
    print(f"\nToken leído correctamente")
    print(f"   Token (primeros 10 caracteres): {token[:10]}...")
    print(f"   Longitud: {len(token)} caracteres")
except Exception as e:
    print(f"\nError leyendo token: {e}")
    exit(1)

# 2. Probar con tipo de cambio (últimos 30 días)
try:
    print(f"\nObteniendo tipo de cambio USD/MXN...")
    print(f"   Serie: SF43718")
    print(f"   Periodo: 2024-11-01 a 2024-11-30")
    
    data = FetchSeriesData(
        'SF43718',
        token,
        '2024-11-01',
        '2024-11-30'
    )
    
    print(f"\nConexión exitosa!")
    print(f"   Registros obtenidos: {len(data)}")
    print(f"\nÚltimos 5 registros:")
    print(data.tail())
    
    print(f"\nEstadísticas:")
    print(f"   Mínimo: ${data['valor'].min():.4f}")
    print(f"   Máximo: ${data['valor'].max():.4f}")
    print(f"   Promedio: ${data['valor'].mean():.4f}")
    
except Exception as e:
    print(f"\nError en la consulta: {e}")
    exit(1)
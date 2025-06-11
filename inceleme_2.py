import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt

# 🌡️ Mevsim katsayıları
season_multipliers = {
    'yaz': 1.3,
    'kış': 1.2,
    'bahar': 1.0,
    'sonbahar': 1.1
}

# 🏙️ İlçe özellikleri (çevresel etki dahil)
districts_template = {
    'Konak':     {'infrastructure_score': 0.8, 'has_critical_infra': True,  'has_backup_power': True,
                  'has_osb': False, 'uses_generators': True,  'is_water_sensitive': True},
    'Bornova':   {'infrastructure_score': 0.7, 'has_critical_infra': True,  'has_backup_power': False,
                  'has_osb': True,  'uses_generators': False, 'is_water_sensitive': True},
    'Karşıyaka': {'infrastructure_score': 0.6, 'has_critical_infra': False, 'has_backup_power': False,
                  'has_osb': False, 'uses_generators': True,  'is_water_sensitive': False},
    'Aliağa':    {'infrastructure_score': 0.9, 'has_critical_infra': True,  'has_backup_power': True,
                  'has_osb': True,  'uses_generators': True,  'is_water_sensitive': False},
    'Tire':      {'infrastructure_score': 0.4, 'has_critical_infra': False, 'has_backup_power': False,
                  'has_osb': False, 'uses_generators': False, 'is_water_sensitive': True},
    'Çeşme':     {'infrastructure_score': 0.5, 'has_critical_infra': False, 'has_backup_power': True,
                  'has_osb': False, 'uses_generators': True,  'is_water_sensitive': True},
}

# 🔢 Öncelik ve toparlanma süresi hesaplama
def compute_priority(d):
    priority = d['infrastructure_score']
    if d['has_critical_infra']:
        priority += 0.2
    return min(priority, 1.0)

def compute_recovery_time(d, season):
    base_time = 60
    if d['infrastructure_score'] < 0.5:
        base_time += 30
    if d['has_backup_power']:
        base_time *= 0.6
    base_time *= season_multipliers[season]
    return int(base_time)

# 🌍 Çevresel etki hesaplama
def compute_environmental_impact(info, duration):
    impact = 0
    if info['has_osb']:
        impact += duration * 2      # üretim kaybı
    if info['uses_generators']:
        impact += duration * 1.5    # hava kirliliği
    if info['is_water_sensitive']:
        impact += duration * 1.2    # su riski
    return round(impact, 1)

# ⚙️ Kesinti işlemi
def power_outage(env, name, info, grid, results, season):
    start = env.now
    with grid.request(priority=1.0 - info['priority']) as req:
        yield req
        yield env.timeout(info['recovery_time'])
        end = env.now
        duration = end - start
        env_impact = compute_environmental_impact(info, duration)
        results.append({
            'ilçe': name,
            'süre': duration,
            'mevsim': season,
            'altyapı_skoru': info['infrastructure_score'],
            'kritik_tesis': int(info['has_critical_infra']),
            'yedek_güç': int(info['has_backup_power']),
            'çevresel_etki': env_impact
        })

# 🕒 Gecikmeli başlatıcı
def delayed_outage(env, delay, name, info, grid, results, season):
    yield env.timeout(delay)
    yield env.process(power_outage(env, name, info, grid, results, season))

# ▶️ Simülasyon başlatıcı
def run_simulation(season='yaz'):
    results = []
    districts = {k: v.copy() for k, v in districts_template.items()}
    for d in districts.values():
        d['priority'] = compute_priority(d)
        d['recovery_time'] = compute_recovery_time(d, season)

    env = simpy.Environment()
    grid = simpy.PriorityResource(env, capacity=2)
    for name, info in districts.items():
        delay = random.randint(0, 20)
        env.process(delayed_outage(env, delay, name, info, grid, results, season))

    env.run(until=300)
    return pd.DataFrame(results)

# 🚀 Çalıştır ve göster
df = run_simulation(season='yaz')
df = df.sort_values(by='çevresel_etki', ascending=False)

# 📊 Görselleştir
plt.figure(figsize=(10, 6))
plt.barh(df['ilçe'], df['çevresel_etki'], color='red')
plt.xlabel("Çevresel Etki Skoru")
plt.title("İlçelere Göre Elektrik Kesintisinin Çevresel Etkisi")
plt.tight_layout()
plt.grid(True)
plt.show()

# 🖨️ Sonuçları yazdır
print("\n🔎 Simülasyon Sonuçları:\n")
print(df)

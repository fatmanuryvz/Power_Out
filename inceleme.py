import simpy
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 🌡️ Mevsim etkisi
SEASON = 'yaz'
season_multipliers = {'yaz': 1.3, 'kış': 1.2, 'bahar': 1.0, 'sonbahar': 1.1}

# İlçe verileri
districts = {
    'Konak':     {'infrastructure_score': 0.8, 'has_critical_infra': True,  'has_backup_power': True},
    'Bornova':   {'infrastructure_score': 0.7, 'has_critical_infra': True,  'has_backup_power': False},
    'Karşıyaka': {'infrastructure_score': 0.6, 'has_critical_infra': False, 'has_backup_power': False},
    'Aliağa':    {'infrastructure_score': 0.9, 'has_critical_infra': True,  'has_backup_power': True},
    'Tire':      {'infrastructure_score': 0.4, 'has_critical_infra': False, 'has_backup_power': False},
    'Çeşme':     {'infrastructure_score': 0.5, 'has_critical_infra': False, 'has_backup_power': True},
}

# Hesaplamalar
def compute_priority(d):
    priority = d['infrastructure_score']
    if d['has_critical_infra']:
        priority += 0.2
    return min(priority, 1.0)

def compute_recovery_time(d):
    base_time = 60
    if d['infrastructure_score'] < 0.5:
        base_time += 30
    if d['has_backup_power']:
        base_time *= 0.6
    base_time *= season_multipliers[SEASON]
    return int(base_time)

for name, d in districts.items():
    d['priority'] = compute_priority(d)
    d['recovery_time'] = compute_recovery_time(d)

# ⏱️ Simülasyon olay akışı
timeline = []

def power_outage(env, name, recovery_time, priority, grid):
    timeline.append((env.now, name, 'started'))
    with grid.request(priority=1.0 - priority) as req:
        yield req
        for minute in range(recovery_time):
            yield env.timeout(1)
            timeline.append((env.now, name, f'working {minute+1}/{recovery_time}'))
        timeline.append((env.now, name, 'recovered'))

def delayed_outage(env, delay, name, recovery_time, priority, grid):
    yield env.timeout(delay)
    yield env.process(power_outage(env, name, recovery_time, priority, grid))

def simulate_for_animation():
    env = simpy.Environment()
    grid = simpy.PriorityResource(env, capacity=2)
    for name, info in districts.items():
        delay = random.randint(0, 20)
        env.process(delayed_outage(env, delay, name, info['recovery_time'], info['priority'], grid))
    while env.peek() < 300:
        env.step()

simulate_for_animation()

# 🎥 Matplotlib Animasyon
fig, ax = plt.subplots(figsize=(10, 6))
bars = {name: ax.barh(name, 0, color='grey') for name in districts}
ax.set_xlim(0, 300)
ax.set_title("Canlı Simülasyon: İzmir İlçelerinde Elektrik Kesintileri")
ax.set_xlabel("Zaman (dakika)")
ax.set_ylabel("İlçeler")
current_time = 0

# İlçe isimlerine karşılık gelen metin etiketleri
text_labels = {name: ax.text(0, i, "", va='center', ha='left', fontsize=9, color='black') for i, name in enumerate(districts)}

def update(frame):
    global current_time
    current_time += 1
    ax.set_xlim(0, current_time + 10)
    for i, ilce in enumerate(districts):
        events = [e for e in timeline if e[1] == ilce and e[0] <= current_time]
        if not events:
            bars[ilce][0].set_width(0)
            bars[ilce][0].set_color('grey')
            text_labels[ilce].set_text("")
            continue

        start_event = next((e for e in events if e[2] == 'started'), None)
        end_event = next((e for e in events if e[2] == 'recovered'), None)

        if not start_event:
            bars[ilce][0].set_width(0)
            bars[ilce][0].set_color('grey')
            text_labels[ilce].set_text("")
            continue

        start_time = start_event[0]

        if end_event:
            end_time = end_event[0]
            duration = end_time - start_time
            bars[ilce][0].set_width(duration)
            bars[ilce][0].set_color('green')
            text_labels[ilce].set_text(f"{duration:.0f} dk")
            text_labels[ilce].set_x(duration + 2)
        else:
            duration = current_time - start_time
            bars[ilce][0].set_width(duration)
            last_event = events[-1][2]
            if last_event == 'started':
                bars[ilce][0].set_color('red')
            elif last_event.startswith('working'):
                bars[ilce][0].set_color('orange')
            text_labels[ilce].set_text(f"{duration:.0f} dk")
            text_labels[ilce].set_x(duration + 2)

    return list(bars.values()) + list(text_labels.values())



ani = FuncAnimation(fig, update, frames=range(1, 301), interval=100, repeat=False)
plt.tight_layout()
plt.show()

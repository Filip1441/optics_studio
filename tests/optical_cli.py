import LightPipes as lp
import numpy as np
import matplotlib.pyplot as plt

def get_float(prompt):
    while True:
        try:
            val = input(prompt)
            if not val: return 0.0
            return float(val)
        except ValueError:
            print("Błąd: Podaj poprawną liczbę.")

class OpticalSimulation:
    def __init__(self):
        self.wvl = 532e-9
        self.detector_size = 0.015
        self.N = 2048
        self.source = None
        self.actions = []
        self.zoom = 100.0
        self.log_scale = False
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def get_nyquist_limit(self, angle_deg):
        """Wylicza maksymalny bezpieczny rozmiar siatki dla danego kąta (Nyquist)"""
        theta = np.radians(angle_deg / 20.0) # Margines bezpieczeństwa
        if theta <= 0: return 1.0 # Brak limitu
        return (self.wvl * self.N) / (2 * np.sin(theta))

    def update_plot(self):
        if self.source is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "BRAK ŹRÓDŁA\nWybierz opcję 0", ha='center', va='center', transform=self.ax.transAxes)
            plt.draw(); plt.pause(0.1); return

        # 1. Startujemy z małej siatki idealnej dla źródła
        curr_l = 0.001 # Start 1mm
        if self.source['type'] == 'point':
            # Jeśli znamy kąt, startujemy z siatki Nyquista dla tego kąta
            angle = self.source.get('angle', 0.1)
            nyq_l = (self.wvl * self.N) / (2 * np.sin(np.radians(angle/2.0)))
            curr_l = min(0.005, nyq_l * 0.5) 
        
        F = lp.Begin(curr_l, self.wvl, self.N)
        
        # Aplikacja źródła
        if self.source['type'] == 'point':
            if self.source['ap_type'] == 'gauss':
                F = lp.GaussAperture(self.source['ap_size'], 0, 0, 1, F)
            else:
                F = lp.CircAperture(self.source['ap_size'], 0, 0, F)
            current_div = self.source.get('angle', 0.1)
        else:
            F = lp.RectAperture(self.source['ap_w'], self.source['ap_h'], 0, 0, 0, F)
            current_div = 0.001 # Fala płaska ma małą dywergencję
        
        current_z = 0.0

        # Wykonywanie akcji z Dynamicznym Resizingiem
        for name, p in self.actions:
            if name == "prop":
                z = p['z']
                # Apodyzacja (miękkie brzegi): likwiduje kwadratowe artefakty od krawędzi siatki
                F = lp.GaussAperture(curr_l*0.48, 0, 0, 10, F) # Super-Gauss 10 rzędu
                
                beam_width = 2 * (current_z + z) * np.tan(np.radians(current_div/2.0))
                # Nowy rozmiar siatki: co najmniej 2x beam_width, ale nie więcej niż Nyquist
                new_l = max(curr_l, beam_width * 2.5)
                nyq_l = (self.wvl * self.N) / (2 * np.sin(np.radians(current_div/2.0)))
                new_l = min(new_l, nyq_l * 0.9)
                
                # Tylko jeśli zmiana jest istotna (>5%)
                if abs(new_l - curr_l) / curr_l > 0.05:
                    F = lp.Interpol(new_l, self.N, 0, 0, 0, F)
                    curr_l = new_l
                
                F = lp.Forvard(z, F)
                current_z += z
                
            elif name == "lens":
                F = lp.Lens(p['f'], p['x'], p['y'], F)
                if p['r'] > 0: F = lp.CircAperture(p['r'], p['x'], p['y'], F)
                if p['f'] != 0:
                    current_div = abs(current_div - (curr_l / p['f']) * (180/np.pi))
            
            elif name == "circ": F = lp.CircAperture(p['r'], p['x'], p['y'], F)
            elif name == "rect": F = lp.RectAperture(p['w'], p['h'], p['x'], p['y'], 0, F)
            elif name == "screen": F = lp.CircScreen(p['r'], p['x'], p['y'], F) # NOWOŚĆ: Przesłona środkowa
            elif name == "grating":
                # OPTYMALIZACJA: Broadcasting zamiast meshgrid
                x = np.linspace(-curr_l/2, curr_l/2, self.N)
                mask_v = ((x % p['p']) < (p['p'] * p['dc'])).astype(float)
                if p['type'] == 'ronchi_lin': F.field = F.field * mask_v[None, :]
                elif p['type'] == 'ronchi_grid': F.field = F.field * mask_v[None, :] * mask_v[:, None]
                elif p['type'] == 'cos_lin':
                    mask = 0.5 * (1 + np.cos(2 * np.pi * x / p['p']))
                    F.field = F.field * mask[None, :]
                elif p['type'] == 'cos_grid':
                    mask = 0.5 * (1 + np.cos(2 * np.pi * x / p['p']))
                    F.field = F.field * mask[None, :] * mask[:, None]
            
            elif name == "testchart":
                # Rysowanie kształtu testowego (Strzałka, F, Krzyż)
                x = np.linspace(-curr_l/2, curr_l/2, self.N)
                X, Y = np.meshgrid(x, x)
                mask = np.zeros_like(X, dtype=bool)
                s = p['size']
                if p['type'] == 'arrow':
                    # Trzon
                    mask |= (np.abs(X) < s/10) & (Y > -s/2) & (Y < s/4)
                    # Grot
                    mask |= (Y > s/4) & (Y < s/2) & (np.abs(X) < (s/2 - Y)*0.8)
                elif p['type'] == 'f':
                    mask |= (X > -s/4) & (X < -s/8) & (np.abs(Y) < s/2) # Pion
                    mask |= (Y > s/4) & (Y < s/2) & (X > -s/8) & (X < s/4) # Góra
                    mask |= (np.abs(Y) < s/10) & (X > -s/8) & (X < s/8)    # Środek
                elif p['type'] == 'cross':
                    mask |= (np.abs(X) < s/20) & (np.abs(Y) < s/2)
                    mask |= (np.abs(Y) < s/20) & (np.abs(X) < s/2)
                F.field = F.field * mask.astype(float)

        # Finalna interpolacja do rozmiaru detektora
        F = lp.Interpol(self.detector_size, self.N, 0, 0, 0, F)
        
        intensity = lp.Intensity(0, F)
        
        # OPTYMALIZACJA PLOTU: Downsampling do 1024 dla szybkości UI
        if self.N > 1024:
            step = self.N // 1024
            intensity = intensity[::step, ::step]

        if self.log_scale: intensity = np.log10(intensity + 1e-9)
            
        self.ax.clear()
        ext = self.detector_size * 1000 / 2
        self.ax.imshow(intensity, cmap='hot', extent=[-ext, ext, -ext, ext])
        
        view_limit = ext * (self.zoom / 100.0)
        self.ax.set_xlim(-view_limit, view_limit)
        self.ax.set_ylim(-view_limit, view_limit)
        
        self.ax.set_title(f"DETEKTOR: {self.detector_size*1000:.1f}mm | SmartGrid L_int: {curr_l*1000:.1f}mm\nKąt wewn: {current_div:.2f} deg | Akcje: {len(self.actions)}")
        plt.draw(); plt.pause(0.01)

    def run(self):
        print("\n" + "="*40)
        print("   LIGHTPIPES LIVE PRO v3.0 [SmartGrid]")
        print("="*40)
        self.detector_size = get_float("Rozmiar DETEKTORA (okna) [m] (np. 0.02): ")
        self.update_plot()

        while True:
            print("\n" + "-"*40)
            src_status = "USTAWIONE" if self.source else "BRAK"
            print(f"ŹRÓDŁO: {src_status} | OKNO PODGLĄDU: {self.detector_size*1000:.1f} mm")
            print("0: WSTAW ŹRÓDŁO      1: Apertura Kołowa    2: Apertura Prost.")
            print("3: PRZESŁONA ŚRODK.   4: Soczewka           5: Propagacja (Z)")
            print("6: Siatka (Grating)   7: COFNIJ (Undo)      8: RESET (Czyść)")
            print("9: ZAKOŃCZ           10: ZOOM (%)         11: SKALA LOGARYTM.")
            print("12: ZMIEŃ ROZMIAR OKNA 13: TEST-CHART (F, Strzałka...)")
            print("-"*40)
            
            choice = input("Wybór: ")
            if choice == '0':
                if self.source: print("Źródło już istnieje! Użyj RESET."); continue
                print("\n1. Płaska, 2. Sferyczna"); s_choice = input("Wybór: ")
                if s_choice == '1':
                    w = get_float("  w [m]: "); h = get_float("  h [m]: ")
                    self.source = {'type': 'plane', 'ap_w': w, 'ap_h': h}
                elif s_choice == '2':
                    print("  Def: 1. Pinhole [m], 2. Kąt [deg]")
                    def_c = input("  Wybór: ")
                    if def_c == '2':
                        ang = get_float("  Kąt [deg]: "); sz = self.wvl / (np.pi * np.radians(ang/2.0))
                    else:
                        sz = get_float("  Pinhole [m]: "); ang = np.degrees(self.wvl / (np.pi * sz)) * 2.0
                    print("  Ap: 1. Kołowa, 2. Gauss"); a_t = input("  Wybór: ")
                    self.source = {'type': 'point', 'ap_type': 'gauss' if a_t=='2' else 'circ', 'ap_size': sz, 'angle': ang}
            
            elif choice == '1': self.actions.append(("circ", {'r': get_float("  r: "), 'x': 0, 'y': 0}))
            elif choice == '2': self.actions.append(("rect", {'w': get_float("  w: "), 'h': get_float("  h: "), 'x': 0, 'y': 0}))
            elif choice == '3': self.actions.append(("screen", {'r': get_float("  r: "), 'x': 0, 'y': 0}))
            elif choice == '4':
                f = get_float("  f: "); r = get_float("  ap: ")
                self.actions.append(("lens", {'f': f, 'r': r, 'x': 0, 'y': 0}))
            elif choice == '5': self.actions.append(("prop", {'z': get_float("  z: ")}))
            elif choice == '6':
                print("\n1.Ronchi_L 2.Ronchi_G 3.Cos_L 4.Cos_G"); tc = input("Wybór: ")
                p = get_float("  Pitch: "); dc = get_float("  DC (0.5): ") or 0.5
                tm = {'1':'ronchi_lin', '2':'ronchi_grid', '3':'cos_lin', '4':'cos_grid'}
                self.actions.append(("grating", {'p': p, 'type': tm.get(tc, 'ronchi_lin'), 'dc': dc}))
            elif choice == '7' and self.actions: self.actions.pop()
            elif choice == '8': self.source = None; self.actions = []
            elif choice == '9': break
            elif choice == '10': z = get_float("  Zoom %: "); self.zoom = z if z>0 else 100
            elif choice == '11': self.log_scale = not self.log_scale
            elif choice == '12': s = get_float("  Okno [m]: "); self.detector_size = s if s>0 else self.detector_size
            elif choice == '13':
                print("\n1. Strzałka, 2. Litera F, 3. Krzyż"); tc_c = input("Wybór: ")
                sz = get_float("  Rozmiar obiektu [m]: ")
                tm = {'1':'arrow', '2':'f', '3':'cross'}
                self.actions.append(("testchart", {'type': tm.get(tc_c, 'arrow'), 'size': sz}))
            
            self.update_plot()

if __name__ == "__main__":
    sim = OpticalSimulation()
    sim.run()

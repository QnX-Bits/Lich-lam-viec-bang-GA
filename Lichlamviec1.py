import wx
import wx.grid
import wx.lib.scrolledpanel as scrolled
from simpleai.search import genetic
from kanren import Relation, facts, run, var
import random
import json
import numpy as np
import threading
import os

# -------------------- CONSTANTS --------------------
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
SHIFTS = ['morning', 'afternoon']
SLOT_PER_SHIFT = 16
SLOT_PER_DAY = SLOT_PER_SHIFT * 2
TOTAL_SLOTS = len(DAYS) * SLOT_PER_DAY
MAX_PATIENTS_PER_SHIFT = 3

# -------------------- CORE LOGIC (from original code) --------------------
available_slot = Relation()

def generate_doctor_schedule(doctors):
    for doctor in doctors:
        all_shifts = []
        for day_idx, day in enumerate(DAYS):
            for shift_idx, shift in enumerate(SHIFTS):
                all_shifts.append((day_idx, shift_idx, f"{day}_{shift}"))
        
        off_shift = random.choice(all_shifts)
        doctor['off_shifts'] = [off_shift]
        
        free_slots = []
        off_day_idx, off_shift_idx, _ = off_shift
        
        for day_idx in range(len(DAYS)):
            for shift_idx in range(len(SHIFTS)):
                if day_idx == off_day_idx and shift_idx == off_shift_idx:
                    continue
                
                start_slot = day_idx * SLOT_PER_DAY + shift_idx * SLOT_PER_SHIFT
                for slot_in_shift in range(SLOT_PER_SHIFT):
                    slot_index = start_slot + slot_in_shift
                    free_slots.append(str(slot_index))
        
        doctor['free_slots'] = free_slots

def get_day_shift_slot(index):
    if index < 0 or index >= TOTAL_SLOTS:
        raise ValueError(f"Invalid slot index: {index}")
    
    day_idx = index // SLOT_PER_DAY
    day = DAYS[day_idx]
    remaining = index % SLOT_PER_DAY
    shift = 'morning' if remaining < SLOT_PER_SHIFT else 'afternoon'
    slot = remaining % SLOT_PER_SHIFT
    
    return day, shift, slot

def get_shift_key(slot_int):
    day_idx = slot_int // SLOT_PER_DAY
    shift_idx = (slot_int % SLOT_PER_DAY) // SLOT_PER_SHIFT
    return (day_idx, shift_idx)

def define_doctor_availability(doctors):
    for doc in doctors:
        for free in doc.get('free_slots', []):
            facts(available_slot, (doc['name'], free))

def is_doctor_working(doctor, day_idx, shift_idx):
    off_shifts = doctor.get('off_shifts', [])
    for off_day_idx, off_shift_idx, _ in off_shifts:
        if off_day_idx == day_idx and off_shift_idx == shift_idx:
            return False
    return True

class ScheduleProblem:
    def __init__(self, doctors, patients):
        self.doctors = doctors
        self.patients = patients

    def generate_random_state(self):
        state = []
        for patient in self.patients:
            if patient.get('priority', 0) < 0 or not patient.get('free_slots', []):
                state.append(None)
                continue

            doctor_choices = [doc for doc in self.doctors
                              if doc.get('specialty') == patient.get('specialty')]
            if not doctor_choices:
                state.append(None)
                continue

            valid_slots = []
            for slot in patient.get('free_slots', []):
                try:
                    slot_int = int(slot)
                    day_idx, shift_idx = slot_int // SLOT_PER_DAY, (slot_int % SLOT_PER_DAY) // SLOT_PER_SHIFT
                    
                    for doc in doctor_choices:
                        if is_doctor_working(doc, day_idx, shift_idx) and str(slot) in doc.get('free_slots', []):
                            valid_slots.append((slot, doc['name']))
                except ValueError:
                    continue

            if valid_slots:
                state.append(random.choice(valid_slots))
            else:
                state.append(None)

        return state

    def count_patients_per_shift(self, state):
        shift_count = {}
        for i, assign in enumerate(state):
            if not assign or i >= len(self.patients):
                continue
            slot, doctor = assign
            try:
                slot_int = int(slot)
                shift_key = get_shift_key(slot_int)
                day_idx, shift_idx = shift_key
                doctor_shift_key = (doctor, day_idx, shift_idx)
                shift_count[doctor_shift_key] = shift_count.get(doctor_shift_key, 0) + 1
            except (ValueError, TypeError):
                continue
        return shift_count

    def value(self, state):
        if not state:
            return 0

        score = 0
        doctor_workload = {}
        slot_usage = {}
        shift_count = self.count_patients_per_shift(state)

        for i, assign in enumerate(state):
            if not assign or i >= len(self.patients):
                continue

            patient = self.patients[i]
            slot, doctor = assign

            try:
                slot_int = int(slot)
            except (ValueError, TypeError):
                continue

            duration = patient.get('duration', 15) // 15
            if slot_int + duration > TOTAL_SLOTS:
                continue

            slots_needed = [str(slot_int + j) for j in range(duration)]

            conflict = False
            for s in slots_needed:
                if s in slot_usage:
                    conflict = True
                    break

            if conflict:
                continue

            doc_obj = next((d for d in self.doctors if d.get('name') == doctor), None)
            if not doc_obj:
                continue

            if not all(s in doc_obj.get('free_slots', []) for s in slots_needed):
                continue

            day_idx, shift_idx = slot_int // SLOT_PER_DAY, (slot_int % SLOT_PER_DAY) // SLOT_PER_SHIFT
            if not is_doctor_working(doc_obj, day_idx, shift_idx):
                continue
            
            doctor_shift_key = (doctor, day_idx, shift_idx)
            current_count = shift_count.get(doctor_shift_key, 0)
            if current_count > MAX_PATIENTS_PER_SHIFT:
                score -= 1000
                continue
                
            for s in slots_needed:
                slot_usage[s] = (doctor, i)

            priority = patient.get('priority', 1)
            score += priority * 10 + max(0, 100 - slot_int)
            
            if current_count <= MAX_PATIENTS_PER_SHIFT:
                score += 50

            doctor_workload[doctor] = doctor_workload.get(doctor, 0) + duration

        workload_values = list(doctor_workload.values())
        imbalance = np.var(workload_values) if len(workload_values) > 1 else 0

        return score - (imbalance * 0.5)

    def mutate(self, state):
        if not state or not self.patients:
            return state

        new_state = list(state)
        for _ in range(10):
            idx = random.randint(0, len(state) - 1)
            patient = self.patients[idx]

            doctor_choices = [doc for doc in self.doctors
                              if doc.get('specialty') == patient.get('specialty')]

            if not doctor_choices or not patient.get('free_slots', []):
                continue

            valid_combinations = []
            for slot in patient.get('free_slots', []):
                try:
                    slot_int = int(slot)
                    day_idx, shift_idx = slot_int // SLOT_PER_DAY, (slot_int % SLOT_PER_DAY) // SLOT_PER_SHIFT
                    
                    for doc in doctor_choices:
                        if str(slot) in doc.get('free_slots', []) and is_doctor_working(doc, day_idx, shift_idx):
                            temp_state = list(new_state)
                            temp_state[idx] = (slot, doc['name'])
                            shift_count = self.count_patients_per_shift(temp_state)
                            doctor_shift_key = (doc['name'], day_idx, shift_idx)
                            if shift_count.get(doctor_shift_key, 0) <= MAX_PATIENTS_PER_SHIFT:
                                valid_combinations.append((slot, doc['name']))
                except (ValueError, TypeError):
                    continue

            if valid_combinations:
                new_state[idx] = random.choice(valid_combinations)
                return new_state

        return new_state

    def crossover(self, s1, s2):
        if not s1 or not s2:
            return s1 or s2 or []

        if len(s1) != len(s2):
            min_len = min(len(s1), len(s2))
            s1 = s1[:min_len]
            s2 = s2[:min_len]

        cut = random.randint(1, len(s1) - 1) if len(s1) > 1 else 1
        return s1[:cut] + s2[cut:]

# -------------------- wxPython GUI --------------------

class ScheduleGrid(wx.grid.Grid):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Create grid structure
        self.CreateGrid(0, 6)  # Start with 0 rows, 6 columns
        
        # Set column labels
        self.SetColLabelValue(0, "Ngày")
        self.SetColLabelValue(1, "Ca")
        self.SetColLabelValue(2, "Bệnh nhân")
        self.SetColLabelValue(3, "Bác sĩ")
        self.SetColLabelValue(4, "Slot")
        self.SetColLabelValue(5, "Ưu tiên")
        
        # Set column widths
        self.SetColSize(0, 80)
        self.SetColSize(1, 80)
        self.SetColSize(2, 100)
        self.SetColSize(3, 120)
        self.SetColSize(4, 60)
        self.SetColSize(5, 80)
        
        # Make grid read-only
        self.EnableEditing(False)

    def update_schedule(self, result, patients):
        # Clear existing data
        if self.GetNumberRows() > 0:
            self.DeleteRows(0, self.GetNumberRows())
        
        schedule_data = []
        for i, assign in enumerate(result.state):
            if assign and i < len(patients):
                slot, doctor = assign
                patient = patients[i]
                try:
                    slot_int = int(slot)
                    day, shift, time_slot = get_day_shift_slot(slot_int)
                    schedule_data.append([
                        day, shift, f"BN{patient.get('id', i+1)}", 
                        doctor, str(time_slot), str(patient.get('priority', 1))
                    ])
                except Exception:
                    continue
        
        # Add rows and populate data
        if schedule_data:
            self.AppendRows(len(schedule_data))
            for row, data in enumerate(schedule_data):
                for col, value in enumerate(data):
                    self.SetCellValue(row, col, value)
                    self.SetCellAlignment(row, col, wx.ALIGN_CENTER, wx.ALIGN_CENTER)

class DoctorWorkloadPanel(scrolled.ScrolledPanel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetupScrolling()
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        
    def update_workload(self, result, doctors, patients):
        # Clear existing content
        self.sizer.Clear(True)
        
        # Calculate doctor shifts
        doctor_shifts = {}
        for i, assign in enumerate(result.state):
            if assign and i < len(patients):
                slot, doctor = assign
                patient = patients[i]
                try:
                    slot_int = int(slot)
                    shift_key = get_shift_key(slot_int)
                    
                    if doctor not in doctor_shifts:
                        doctor_shifts[doctor] = {}
                    if shift_key not in doctor_shifts[doctor]:
                        doctor_shifts[doctor][shift_key] = []
                        
                    doctor_shifts[doctor][shift_key].append(f"BN{patient.get('id', i+1)}")
                except Exception:
                    continue
        
        # Create workload display
        title = wx.StaticText(self, label=f"TỔNG KẾT CÔNG VIỆC CÁC BÁC SĨ (Giới hạn: {MAX_PATIENTS_PER_SHIFT} BN/buổi)")
        title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.sizer.Add(title, 0, wx.ALL, 10)
        
        for doctor in doctors:
            doctor_name = doctor['name']
            
            # Doctor name
            doctor_label = wx.StaticText(self, label=f"Bác sĩ {doctor_name}:")
            doctor_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            self.sizer.Add(doctor_label, 0, wx.LEFT, 20)
            
            # Off shift info
            if doctor.get('off_shifts'):
                off_day_idx, off_shift_idx, _ = doctor['off_shifts'][0]
                off_day = DAYS[off_day_idx]
                off_shift = SHIFTS[off_shift_idx]
                off_text = wx.StaticText(self, label=f"  Nghỉ: {off_day} ({off_shift})")
                self.sizer.Add(off_text, 0, wx.LEFT, 40)
            
            # Work schedule
            total_patients = 0
            for day_idx in range(len(DAYS)):
                for shift_idx in range(len(SHIFTS)):
                    shift_key = (day_idx, shift_idx)
                    day_name = DAYS[day_idx]
                    shift_name = SHIFTS[shift_idx]
                    
                    if not is_doctor_working(doctor, day_idx, shift_idx):
                        continue
                        
                    if doctor_name in doctor_shifts and shift_key in doctor_shifts[doctor_name]:
                        patients_list = doctor_shifts[doctor_name][shift_key]
                        count = len(patients_list)
                        total_patients += count
                        status = "⚠️ VƯỢT" if count > MAX_PATIENTS_PER_SHIFT else "✅ OK"
                        
                        shift_text = wx.StaticText(self, 
                            label=f"  {day_name} ({shift_name}): {count}/{MAX_PATIENTS_PER_SHIFT} BN {status} - {', '.join(patients_list)}")
                        
                        if count > MAX_PATIENTS_PER_SHIFT:
                            shift_text.SetForegroundColour(wx.Colour(255, 0, 0))  # Red
                        
                        self.sizer.Add(shift_text, 0, wx.LEFT, 40)
                    else:
                        shift_text = wx.StaticText(self, 
                            label=f"  {day_name} ({shift_name}): 0/{MAX_PATIENTS_PER_SHIFT} BN ✅ OK - Rảnh")
                        shift_text.SetForegroundColour(wx.Colour(0, 128, 0))  # Green
                        self.sizer.Add(shift_text, 0, wx.LEFT, 40)
            
            # Total
            total_text = wx.StaticText(self, label=f"  → Tổng: {total_patients} bệnh nhân")
            total_text.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL))
            self.sizer.Add(total_text, 0, wx.LEFT, 40)
            
            self.sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 5)
        
        self.Layout()
        self.SetupScrolling()

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title="Hệ thống lập lịch khám bệnh", size=(1000, 700))
        
        self.doctors = []
        self.patients = []
        self.result = None
        
        self.create_menu()
        self.create_ui()
        self.create_status_bar()
        
        self.Center()

    def create_menu(self):
        menubar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        load_item = file_menu.Append(wx.ID_OPEN, "&Tải dữ liệu\tCtrl+O", "Tải dữ liệu từ file JSON")
        run_item = file_menu.Append(wx.ID_ANY, "&Chạy thuật toán\tCtrl+R", "Chạy thuật toán tối ưu")
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "&Thoát\tCtrl+Q", "Thoát chương trình")
        
        menubar.Append(file_menu, "&File")
        
        # Bind events
        self.Bind(wx.EVT_MENU, self.on_load_data, load_item)
        self.Bind(wx.EVT_MENU, self.on_run_algorithm, run_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        
        self.SetMenuBar(menubar)

    def create_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Control panel
        control_panel = wx.Panel(panel)
        control_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.load_btn = wx.Button(control_panel, label="Tải dữ liệu")
        self.run_btn = wx.Button(control_panel, label="Chạy thuật toán")
        self.run_btn.Enable(False)
        
        control_sizer.Add(self.load_btn, 0, wx.ALL, 5)
        control_sizer.Add(self.run_btn, 0, wx.ALL, 5)
        
        # Progress bar
        self.progress = wx.Gauge(control_panel, range=100)
        control_sizer.Add(self.progress, 1, wx.ALL | wx.EXPAND, 5)
        
        control_panel.SetSizer(control_sizer)
        
        # Notebook for tabs
        self.notebook = wx.Notebook(panel)
        
        # Schedule tab
        schedule_panel = wx.Panel(self.notebook)
        schedule_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.schedule_grid = ScheduleGrid(schedule_panel)
        schedule_sizer.Add(self.schedule_grid, 1, wx.EXPAND | wx.ALL, 5)
        
        schedule_panel.SetSizer(schedule_sizer)
        self.notebook.AddPage(schedule_panel, "Lịch khám")
        
        # Workload tab
        self.workload_panel = DoctorWorkloadPanel(self.notebook)
        self.notebook.AddPage(self.workload_panel, "Công việc bác sĩ")
        
        # Summary tab
        summary_panel = wx.Panel(self.notebook)
        summary_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.summary_text = wx.TextCtrl(summary_panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        summary_sizer.Add(self.summary_text, 1, wx.EXPAND | wx.ALL, 5)
        
        summary_panel.SetSizer(summary_sizer)
        self.notebook.AddPage(summary_panel, "Tóm tắt")
        
        # Add to main sizer
        main_sizer.Add(control_panel, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(main_sizer)
        
        # Bind events
        self.load_btn.Bind(wx.EVT_BUTTON, self.on_load_data)
        self.run_btn.Bind(wx.EVT_BUTTON, self.on_run_algorithm)

    def create_status_bar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("Sẵn sàng")

    def on_load_data(self, event):
        try:
            # Check if files exist
            if not os.path.exists('doctor.json') or not os.path.exists('patin.json'):
                wx.MessageBox("Không tìm thấy file doctor.json hoặc patin.json", 
                             "Lỗi", wx.OK | wx.ICON_ERROR)
                return
            
            with open('doctor.json', 'r', encoding='utf-8') as f:
                self.doctors = json.load(f)
            with open('patin.json', 'r', encoding='utf-8') as f:
                self.patients = json.load(f)
            
            # Generate doctor schedules
            generate_doctor_schedule(self.doctors)
            
            self.run_btn.Enable(True)
            self.statusbar.SetStatusText(f"Đã tải {len(self.doctors)} bác sĩ và {len(self.patients)} bệnh nhân")
            
            # Update summary
            summary = f"=== THÔNG TIN DỮ LIỆU ===\n"
            summary += f"Số bác sĩ: {len(self.doctors)}\n"
            summary += f"Số bệnh nhân: {len(self.patients)}\n\n"
            summary += "=== LỊCH NGHỈ CÁC BÁC SĨ ===\n"
            
            for doctor in self.doctors:
                if doctor.get('off_shifts'):
                    off_day_idx, off_shift_idx, _ = doctor['off_shifts'][0]
                    off_day_name = DAYS[off_day_idx]
                    off_shift_name = SHIFTS[off_shift_idx]
                    summary += f"Bác sĩ {doctor['name']} nghỉ: {off_day_name} ({off_shift_name})\n"
            
            self.summary_text.SetValue(summary)
            
        except Exception as e:
            wx.MessageBox(f"Lỗi khi tải dữ liệu: {str(e)}", "Lỗi", wx.OK | wx.ICON_ERROR)

    def on_run_algorithm(self, event):
        if not self.doctors or not self.patients:
            wx.MessageBox("Vui lòng tải dữ liệu trước", "Thông báo", wx.OK | wx.ICON_WARNING)
            return
        
        # Disable button during processing
        self.run_btn.Enable(False)
        self.statusbar.SetStatusText("Đang chạy thuật toán...")
        
        # Run algorithm in separate thread
        thread = threading.Thread(target=self.run_genetic_algorithm)
        thread.daemon = True
        thread.start()

    def run_genetic_algorithm(self):
        try:
            define_doctor_availability(self.doctors)
            problem = ScheduleProblem(self.doctors, self.patients)
            
            # Update progress periodically
            wx.CallAfter(self.update_progress, 50)
            
            self.result = genetic(
                problem=problem,
                population_size=100,
                mutation_chance=0.3,
                iterations_limit=300
            )
            
            wx.CallAfter(self.update_progress, 100)
            wx.CallAfter(self.algorithm_completed)
            
        except Exception as e:
            wx.CallAfter(self.algorithm_error, str(e))

    def update_progress(self, value):
        self.progress.SetValue(value)

    def algorithm_completed(self):
        # Update UI with results
        self.schedule_grid.update_schedule(self.result, self.patients)
        self.workload_panel.update_workload(self.result, self.doctors, self.patients)
        
        # Update summary
        assigned_count = sum(1 for assign in self.result.state if assign)
        score = ScheduleProblem(self.doctors, self.patients).value(self.result.state)
        
        summary = self.summary_text.GetValue()
        summary += f"\n\n=== KẾT QUẢ THUẬT TOÁN ===\n"
        summary += f"Tổng số bệnh nhân được lịch hẹn: {assigned_count}/{len(self.patients)}\n"
        summary += f"Điểm số của lịch: {score}\n"
        
        self.summary_text.SetValue(summary)
        
        # Re-enable button and update status
        self.run_btn.Enable(True)
        self.progress.SetValue(0)
        self.statusbar.SetStatusText(f"Hoàn thành - {assigned_count}/{len(self.patients)} bệnh nhân được xếp lịch")

    def algorithm_error(self, error_msg):
        wx.MessageBox(f"Lỗi khi chạy thuật toán: {error_msg}", "Lỗi", wx.OK | wx.ICON_ERROR)
        self.run_btn.Enable(True)
        self.progress.SetValue(0)
        self.statusbar.SetStatusText("Sẵn sàng")

    def on_exit(self, event):
        self.Close()

class MedicalSchedulerApp(wx.App):
    def OnInit(self):
        frame = MainFrame()
        frame.Show()
        return True

if __name__ == '__main__':
    app = MedicalSchedulerApp()
    app.MainLoop()
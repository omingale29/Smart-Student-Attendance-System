import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import os
import face_recognition
import pickle
import csv
import numpy as np
from datetime import datetime
import threading

try:
    import winsound
except ImportError:
    winsound = None

# ================= CONFIG =================
DATASET_DIR = "dataset"
ENCODING_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"
CURRENT_SLOT_FILE = "current_slot.txt"

os.makedirs(DATASET_DIR, exist_ok=True)

# ================= UTILITIES =================
def load_encodings():
    if not os.path.exists(ENCODING_FILE):
        return [], []
    with open(ENCODING_FILE, "rb") as f:
        return pickle.load(f)


def save_encodings(encodings, names):
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump((encodings, names), f)


def get_current_slot():
    today = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(CURRENT_SLOT_FILE):
        with open(CURRENT_SLOT_FILE, "w") as f:
            f.write(f"{today},1")
        return 1

    try:
        with open(CURRENT_SLOT_FILE, "r") as f:
            saved_date, slot = f.read().strip().split(",")

        if saved_date != today:
            with open(CURRENT_SLOT_FILE, "w") as f:
                f.write(f"{today},1")
            return 1

        return int(slot)

    except Exception:
        # ✅ FIXED ERROR HERE
        with open(CURRENT_SLOT_FILE, "w") as f:
            f.write(f"{today},1")
        return 1


def next_lecture_slot():
    slot = get_current_slot() + 1
    today = datetime.now().strftime("%Y-%m-%d")
    with open(CURRENT_SLOT_FILE, "w") as f:
        f.write(f"{today},{slot}")
    update_status_bar()
    messagebox.showinfo("Next Slot", f"Moved to lecture slot {slot}")


def get_student_count():
    _, names = load_encodings()
    return len(names)


def get_today_present_count():
    if not os.path.exists(ATTENDANCE_FILE):
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    slot = get_current_slot()
    seen = set()

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[1] == today and int(row[2]) == slot:
                seen.add(row[0])

    return len(seen)


def update_status_bar():
    total = get_student_count()
    present = get_today_present_count()
    absent = max(0, total - present)
    slot = get_current_slot()
    status_label.config(
        text=f"🎓 Slot: {slot} | 👤 Registered: {total} | ✅ Present: {present} | ❌ Absent: {absent}"
    )

# ================= REGISTER =================
def register_face():
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Enter student name")
        return

    known_enc, known_names = load_encodings()
    if name in known_names:
        messagebox.showwarning("Warning", f"{name} already registered")
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Error", "Camera not found")
        return

    user_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    images = []
    count = 0
    messagebox.showinfo("Info", "Press S to capture 5 images, ESC to cancel")

    while count < 5:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Captured: {count}/5 | Press S",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cam.release()
            cv2.destroyAllWindows()
            return

        if key == ord('s'):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb)
            if not faces:
                messagebox.showwarning("Warning", "No face detected")
                continue

            count += 1
            img_path = os.path.join(user_dir, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            images.append(img_path)

    cam.release()
    cv2.destroyAllWindows()

    enc_list = []
    for img_path in images:
        img = cv2.imread(img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if enc:
            enc_list.append(enc[0])

    if not enc_list:
        messagebox.showerror("Error", "Face encoding failed")
        return

    known_enc.append(np.mean(enc_list, axis=0))
    known_names.append(name)
    save_encodings(known_enc, known_names)
    update_status_bar()
    messagebox.showinfo("Success", f"{name} registered successfully")

# ================= DELETE =================
def delete_student():
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Enter student name")
        return

    known_enc, known_names = load_encodings()
    if name not in known_names:
        messagebox.showerror("Error", "Student not found")
        return

    idx = known_names.index(name)
    known_names.pop(idx)
    known_enc.pop(idx)
    save_encodings(known_enc, known_names)

    user_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(user_dir):
        for file in os.listdir(user_dir):
            os.remove(os.path.join(user_dir, file))
        os.rmdir(user_dir)

    update_status_bar()
    messagebox.showinfo("Deleted", f"{name} removed successfully")

# ================= ATTENDANCE =================
def mark_attendance():
    known_enc, known_names = load_encodings()
    if not known_enc:
        messagebox.showerror("Error", "No registered students")
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Error", "Camera not found")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    beep_cooldown = 0

    try:
        while True:
            current_slot = get_current_slot()
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            unknown_found = False

            for enc, loc in zip(encodings, locations):
                distances = face_recognition.face_distance(known_enc, enc)
                best = np.argmin(distances)
                confidence = (1 - distances[best]) * 100
                top, right, bottom, left = loc

                if distances[best] < 0.50:
                    name = known_names[best]
                    already_marked = False

                    if os.path.exists(ATTENDANCE_FILE):
                        with open(ATTENDANCE_FILE, "r", newline="") as f:
                            reader = csv.reader(f)
                            next(reader, None)
                            for row in reader:
                                if len(row) >= 3 and row[0] == name and row[1] == today and int(row[2]) == current_slot:
                                    already_marked = True
                                    break

                    if not already_marked:
                        file_exists = os.path.exists(ATTENDANCE_FILE)
                        with open(ATTENDANCE_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["Name", "Date", "Slot", "Time", "Confidence(%)"])
                            writer.writerow([name, today, current_slot,
                                             datetime.now().strftime("%H:%M:%S"),
                                             f"{confidence:.2f}"])

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.1f}%)",
                                (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    unknown_found = True
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown",
                                (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if unknown_found and winsound and beep_cooldown == 0:
                winsound.Beep(1000, 300)
                beep_cooldown = 30

            if beep_cooldown > 0:
                beep_cooldown -= 1

            cv2.putText(frame, f"Lecture Slot: {current_slot}",
                        (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Present: {get_today_present_count()}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("AI Lecture-wise Attendance", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        update_status_bar()


def start_attendance():
    threading.Thread(target=mark_attendance, daemon=True).start()

# ================= VIEW =================
def view_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        messagebox.showinfo("Info", "No attendance records")
        return

    win = tk.Toplevel(root)
    win.title("Attendance Records")
    win.geometry("700x400")

    tree = ttk.Treeview(win, columns=("Name", "Date", "Slot", "Time", "Confidence"), show="headings")
    for col in ("Name", "Date", "Slot", "Time", "Confidence"):
        tree.heading(col, text=col)
        tree.column(col, width=130)
    tree.pack(fill="both", expand=True)

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            tree.insert("", "end", values=row)

# ================= GUI =================
root = tk.Tk()
root.title("AI Lecture-wise Attendance System")
root.geometry("540x520")
root.configure(bg="#1e1e2f")
root.resizable(False, False)

header = tk.Label(root, text="AI Lecture-wise Attendance System",
                  bg="#2a2a40", fg="white",
                  font=("Segoe UI", 14, "bold"), pady=15)
header.pack(fill="x")

status_label = tk.Label(root, bg="#12122a", fg="#00e676",
                        font=("Segoe UI", 9, "bold"), pady=8)
status_label.pack(fill="x")

frame = tk.Frame(root, bg="#1e1e2f")
frame.pack(pady=20)

tk.Label(frame, text="Enter Student Name", bg="#1e1e2f", fg="white",
         font=("Segoe UI", 11)).pack(pady=5)

name_entry = tk.Entry(frame, width=28, font=("Segoe UI", 11), justify="center")
name_entry.pack(pady=10, ipady=6)

btn_frame = tk.Frame(frame, bg="#1e1e2f")
btn_frame.pack()

tk.Button(btn_frame, text="📷 Register Face", width=20, bg="#2196F3", fg="white",
          font=("Segoe UI", 10, "bold"), command=register_face).grid(row=0, column=0, padx=5, pady=5)

tk.Button(btn_frame, text="✅ Start Attendance", width=20, bg="#4CAF50", fg="white",
          font=("Segoe UI", 10, "bold"), command=start_attendance).grid(row=0, column=1, padx=5, pady=5)

tk.Button(btn_frame, text="📋 View Attendance", width=20, bg="#9C27B0", fg="white",
          font=("Segoe UI", 10, "bold"), command=view_attendance).grid(row=1, column=0, padx=5, pady=5)

tk.Button(btn_frame, text="🗑 Delete Student", width=20, bg="#f44336", fg="white",
          font=("Segoe UI", 10, "bold"), command=delete_student).grid(row=1, column=1, padx=5, pady=5)

tk.Button(btn_frame, text="⏭ Next Lecture Slot", width=42, bg="#ff9800", fg="white",
          font=("Segoe UI", 10, "bold"), command=next_lecture_slot).grid(row=2, column=0, columnspan=2, pady=8)

update_status_bar()
root.mainloop()
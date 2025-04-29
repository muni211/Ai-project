# File: graphics/ui_engine.py
# ממשק משתמש פשוט באמצעות Tkinter שמשתמש במודול NLP

import tkinter as tk
from tkinter import messagebox
from nlp.nlp_module import process_text

def analyze_text():
    """ מקבל טקסט מהמשתמש, מעבד אותו עם NLP ומציג תוצאה. """
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("שגיאה", "נא להזין טקסט!")
        return

    processed_tokens = process_text(user_input)
    result_text.set("Processed words:\n" + ", ".join(processed_tokens))

# יצירת חלון ראשי
root = tk.Tk()
root.title("מערכת AI - NLP")
root.geometry("400x250")

# תווית טקסט
label = tk.Label(root, text="הזן טקסט לניתוח:", font=("Arial", 14))
label.pack(pady=10)

# תיבת קלט מהמשתמש
entry = tk.Entry(root, width=40)
entry.pack(pady=5)

# כפתור לעיבוד הטקסט
button = tk.Button(root, text="Analyze Text", command=analyze_text)
button.pack(pady=10)

# תיבת תוצאה
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 12), wraplength=350)
result_label.pack(pady=10)

# הפעלת החלון
root.mainloop()

def backup_reminder(title, points, popup=True):
    """Show a prominent BACKUP reminder in the notebook, and (best effort) an OS pop-up.

    Args:
        title: short reminder title, e.g. "Step 5 - Back up the Sorting folder".
        points: list of strings, shown as bullet points.
        popup: if True, also try to open a small OS pop-up window. This is best effort:
            it is silently skipped if there is no display / tkinter available, so it
            never breaks the notebook.
    """
    # 1) Always show a styled box inline in the notebook output.
    try:
        from IPython.display import HTML, display

        bullets = "".join(f"<li style='margin:3px 0;'>{p}</li>" for p in points)
        display(
            HTML(
                "<div style='border:2px solid #e0a100;background:#fff8e1;border-radius:8px;"
                "padding:10px 14px;margin:6px 0;font-size:14px;color:#5c4400;'>"
                f"<b>&#128190; BACKUP REMINDER &mdash; {title}</b>"
                f"<ul style='margin:6px 0 0 18px;'>{bullets}</ul></div>"
            )
        )
    except Exception:
        print(f"\n===== BACKUP REMINDER - {title} =====")
        for p in points:
            print(f"  - {p}")
        print("=" * (len(title) + 26))

    # 2) Optionally raise a real OS pop-up window (forces acknowledgement).
    if popup:
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showinfo(
                f"Backup reminder - {title}", "\n".join(f"• {p}" for p in points)
            )
            root.destroy()
        except Exception:
            pass  # no display / tkinter -> the inline box above is enough

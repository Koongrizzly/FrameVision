helpers/README_visualizer_catalog.txt

What this is
------------
A drop‑in registry of 10 eye‑candy visualizers. It lets your existing selector
show these new options without hard‑coding strings all over the UI.

Files in this patch
-------------------
- helpers/visualizer_catalog.py  (the registry)
- helpers/README_visualizer_catalog.txt (this file)

How to wire into your selector (2 tiny changes)
-----------------------------------------------
1) Populate your QComboBox from the catalog:

    from helpers.visualizer_catalog import to_combo_items

    combo_visuals.clear()
    for name, key in to_combo_items():
        combo_visuals.addItem(name, userData=key)

2) When the user clicks "Run" (or similar), resolve the selected spec:

    from helpers.visualizer_catalog import find_by_key
    key = combo_visuals.currentData()  # the userData you added
    spec = find_by_key(key)
    # 'spec' is a dict with:
    #   key, name, short, engine, req (pip packages), audio_map, defaults
    # Pass 'spec' to your visualizer launcher / renderer.

Notes
-----
- The catalog doesn’t implement the renderers; it standardizes names and
  parameters so you can plug into your existing engine. Keep your current
  three visuals as‑is; just append these.
- If you want me to patch the exact UI file that holds the visual list,
  send that file and I’ll ship a combo ZIP that auto‑wires the list.

License
-------
Public domain (do whatever you like).

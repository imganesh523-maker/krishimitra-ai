"""Take a screenshot of the running Streamlit app with headless Chromium (Playwright).
Start the app first:  streamlit run app.py --server.headless true --server.port 8501
"""
import sys
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
out = sys.argv[2] if len(sys.argv) > 2 else "assets/app_screenshot.png"
sample_index = int(sys.argv[3]) if len(sys.argv) > 3 else 1
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1100})
    pg.goto(url); pg.wait_for_selector("text=KrishiMitra", timeout=60000)
    pg.wait_for_timeout(2000)
    if len(sys.argv) > 4 and sys.argv[4] == "hi":  # switch UI to Hindi first
        pg.locator("[data-testid='stRadioOption']").nth(1).click(); pg.wait_for_timeout(3000)
    # open the sample selectbox in the sidebar and choose a sample
    pg.locator("[data-testid='stSelectbox']").first.click()
    pg.wait_for_timeout(800)
    pg.locator("[role='option']").nth(sample_index).click()
    pg.wait_for_selector("text=Grad-CAM", timeout=90000)
    pg.wait_for_timeout(5000); pg.get_by_role("tab").nth(2).click()  # show the Treatment tab
    pg.wait_for_timeout(6000)
    pg.screenshot(path=out, full_page=True)
    b.close()
print("saved", out)

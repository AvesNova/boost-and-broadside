// Renders whitepaper.html to whitepaper.pdf with headless Chromium.
// Usage: node docs/portfolio/build.js   (needs the `playwright` package)
//
// The committed PDF carries no email address. For a copy to send with an
// application, put one in the byline and write it somewhere outside the repo:
//   CONTACT_EMAIL=you@example.com OUT=/tmp/whitepaper.pdf node docs/portfolio/build.js
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const dir = __dirname;
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.join(dir, "whitepaper.html"), { waitUntil: "networkidle" });
  const email = process.env.CONTACT_EMAIL;
  if (email) {
    await page.evaluate((addr) => {
      const slot = document.getElementById("contact-email");
      const link = document.createElement("a");
      link.href = "mailto:" + addr;
      link.textContent = addr;
      slot.append(link, "\u00a0·\u00a0\u00a0");
    }, email);
  }
  await page.pdf({
    path: process.env.OUT || path.join(dir, "whitepaper.pdf"),
    format: "Letter",
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate:
      '<div style="width:100%;font-family:Inter,sans-serif;font-size:7.5pt;color:#5b6676;' +
      'padding:0 0.85in;display:flex;justify-content:space-between;">' +
      "<span>Boost and Broadside · v1 results</span>" +
      '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
  });
  await browser.close();
})();

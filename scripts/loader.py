from crawl4ai import AsyncWebCrawler
import asyncio
import pandas as pd

df = pd.read_csv("./heum_target_urls_20260314011945.csv")

def _get_page_content(result) -> str:
    """CrawlResult에서 본문 텍스트 추출. markdown 우선, 없으면 cleaned_html/html."""
    if not result:
        return ""
    md = getattr(result, "markdown", None)
    if md is not None:
        if hasattr(md, "raw_markdown"):
            return md.raw_markdown or ""
        if isinstance(md, str):
            return md
    return getattr(result, "cleaned_html", "") or getattr(result, "html", "") or ""


async def save_pages_to_md(urls, output_dir="./md_pages"):
    import os
    import re
    os.makedirs(output_dir, exist_ok=True)
    async with AsyncWebCrawler(verbose=False) as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url)
                if not getattr(result, "success", False):
                    continue
                content = _get_page_content(result)
                if not content:
                    continue
                # 파일명 안전하게
                basename = re.sub(r"[^a-zA-Z0-9]", "_", url)
                filename = f"{output_dir}/{basename}.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"# {url}\n\n")
                    f.write(content)
                print(f"✅ Saved: {filename}")
            except Exception as e:
                print(f"❌ Error for {url}: {e}")

urls = df['url'].tolist()
asyncio.run(save_pages_to_md(urls))
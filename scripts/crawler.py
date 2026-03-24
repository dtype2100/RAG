import asyncio
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from urllib.parse import urlparse, urljoin
from urllib.request import Request, urlopen
import nest_asyncio
from crawl4ai import AsyncWebCrawler

# CrawlerRunConfig는 버전에 따라 없을 수 있음 → 예외 시 폴백
try:
    from crawl4ai import CrawlerRunConfig
    HAS_CRAWLER_RUN_CONFIG = True
except ImportError:
    HAS_CRAWLER_RUN_CONFIG = False

# 주피터 및 윈도우 환경 비동기 에러 방지
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()


def _build_headers_for_domain(start_url: str) -> dict:
    """시작 URL 기준 동일 도메인 Referer 및 공통 헤더 생성."""
    parsed = urlparse(start_url)
    base_origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    return {
        "Referer": base_origin + "/",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    }


def _normalize_and_validate_url(
    href: str,
    current_url: str,
    base_domain: str,
) -> str | None:
    """
    href를 절대 URL로 정규화하고, 동일 도메인·유효 scheme인지 검사해 반환.
    유효하지 않으면 None.
    """
    if not href or not str(href).strip():
        return None
    href = href.strip()
    if href.lower().startswith(("mailto:", "javascript:", "tel:", "#")):
        return None
    absolute = urljoin(current_url, href)
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.netloc or base_domain not in parsed.netloc:
        return None
    return absolute


def _should_skip_url(url_lower: str) -> bool:
    """다운로드/파일/불필요 확장자 등 수집 제외 대상 여부."""
    if url_lower.endswith((".pdf", ".png", ".jpg", ".zip", ".hwp", ".xlsx", ".doc", ".ppt")):
        return True
    if "download" in url_lower or "file" in url_lower or "board/atch" in url_lower:
        return True
    return False


# sitemap 요청 시 서버 차단 방지를 위한 User-Agent (일반 브라우저로 위장)
_SITEMAP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _fetch_sitemap_text_sync(url: str, timeout: int = 15) -> str | None:
    """동기로 sitemap URL 내용 가져오기. User-Agent 헤더로 차단 방지. 실패 시 None."""
    try:
        req = Request(url, headers={"User-Agent": _SITEMAP_USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _extract_locs_from_xml(text: str) -> list[str]:
    """XML 텍스트에서 sitemap <loc> URL 목록 추출."""
    locs = []
    try:
        root = ET.fromstring(text)
        for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
            if elem.text:
                locs.append(elem.text.strip())
        if not locs:
            for elem in root.iter("loc"):
                if elem.text:
                    locs.append(elem.text.strip())
    except Exception:
        pass
    return locs


async def fetch_sitemap_urls(base_url: str) -> set[str]:
    """
    base_url 도메인의 sitemap.xml(/sitemap_index.xml)에서 URL 집합 추출.
    실패 시(404, 파싱 에러, 네트워크 등) 빈 set 반환. aiohttp 없이 stdlib만 사용.
    """
    urls = set()
    parsed = urlparse(base_url)
    base_origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    sitemap_candidates = [f"{base_origin}/sitemap.xml", f"{base_origin}/sitemap_index.xml"]

    try:
        for sitemap_url in sitemap_candidates:
            text = await asyncio.to_thread(_fetch_sitemap_text_sync, sitemap_url)
            if not text:
                continue
            locs = _extract_locs_from_xml(text)
            for loc in locs:
                loc_lower = loc.lower()
                if "sitemap" in loc_lower and loc_lower.endswith(".xml"):
                    continue
                urls.add(loc)
            if urls:
                break
    except Exception:
        pass
    return urls


async def extract_urls_with_crawl4ai(start_url: str, max_pages: int = 500) -> pd.DataFrame:
    """
    crawl4ai로 start_url부터 BFS 탐색해 동일 도메인 내부 링크를 수집해 DataFrame 반환.
    - Referer/헤더는 start_url 기준 동일 도메인 사용.
    - CrawlerRunConfig(wait_for, delay_before_return_html) 사용 시도 후, 실패 시 config 없이 1회 재시도.
    - 링크는 절대 URL로 정규화 후 동일 도메인만 수집하며, dict/str 링크 객체 모두 처리.
    """
    base_domain = urlparse(start_url).netloc
    visited = set()
    queue = [start_url]
    all_urls = set([start_url])
    headers = _build_headers_for_domain(start_url)

    async with AsyncWebCrawler(verbose=False) as crawler:
        print(f"🔍 URL 탐색 시작: {start_url}")

        while queue and len(visited) < max_pages:
            current_url = queue.pop(0)

            if current_url in visited:
                continue

            visited.add(current_url)
            print(f"탐색 중 [{len(visited)}]: {current_url}")

            await asyncio.sleep(1.5)

            result = None
            used_fallback = False

            # 1) CrawlerRunConfig로 시도 (SPA 대기 등)
            if HAS_CRAWLER_RUN_CONFIG:
                try:
                    config = CrawlerRunConfig(
                        wait_for="css:a[href]",
                        delay_before_return_html=2.0,
                        page_timeout=60000,
                        magic=True,
                    )
                    result = await crawler.arun(url=current_url, config=config, headers=headers)
                except (TypeError, AttributeError, Exception) as e:
                    print(f"⚠️ [Config 실패] 폴백 재시도: {current_url} ({e!r})")
                    used_fallback = True
                    result = None

            if result is None:
                try:
                    result = await crawler.arun(
                        url=current_url,
                        magic=True,
                        headers=headers,
                    )
                except Exception as e:
                    print(f"⏩ [스킵] 접속 에러: {current_url} ({e!r})")
                    continue

            if not result or not getattr(result, "success", False):
                if not used_fallback and HAS_CRAWLER_RUN_CONFIG:
                    try:
                        result = await crawler.arun(url=current_url, magic=True, headers=headers)
                    except Exception:
                        pass
                if not result or not getattr(result, "success", False):
                    print(f"⚠️ [실패] 접근 거부 또는 페이지 없음: {current_url}")
                    continue

            links = getattr(result, "links", None) or {}
            internal_links = links.get("internal", []) if isinstance(links, dict) else []

            for link_obj in internal_links:
                href = None
                if isinstance(link_obj, str):
                    href = link_obj
                elif isinstance(link_obj, dict):
                    href = link_obj.get("href")
                if not href:
                    continue

                next_url = _normalize_and_validate_url(href, current_url, base_domain)
                if not next_url:
                    continue
                if _should_skip_url(next_url.lower()):
                    continue
                if next_url not in all_urls:
                    all_urls.add(next_url)
                    queue.append(next_url)

    df = pd.DataFrame(list(all_urls), columns=["url"])
    df = df.sort_values(by="url").reset_index(drop=True)
    return df


async def main():
    target_url = "https://www.heum.ai/ko/home"

    # 1) 크롤로 URL 추출
    df_urls = await extract_urls_with_crawl4ai(target_url, max_pages=200)
    url_set = set(df_urls["url"].tolist())

    # 2) Sitemap에서 URL 병합 (실패 시 무시)
    try:
        sitemap_urls = await fetch_sitemap_urls(target_url)
        if sitemap_urls:
            base_domain = urlparse(target_url).netloc
            for u in sitemap_urls:
                if base_domain in urlparse(u).netloc and not _should_skip_url(u.lower()):
                    url_set.add(u)
            print(f"📄 Sitemap에서 {len(sitemap_urls)}개 URL 추가 반영")
    except Exception as e:
        print(f"📄 Sitemap 수집 생략 (오류: {e!r})")

    df_urls = pd.DataFrame(sorted(url_set), columns=["url"])

    print(f"\n✅ 탐색 완료! 총 {len(df_urls)}개의 고유 URL을 찾았습니다.")

    output_filename = "heum_target_urls"
    import datetime
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    df_urls.to_csv(f"{output_filename}_{now}.csv", index=False, encoding="utf-8-sig")

    # df_urls.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"📁 추출된 URL이 '{output_filename}' 파일로 저장되었습니다. 엑셀에서 확인해 보세요!")


if __name__ == "__main__":
    asyncio.run(main())

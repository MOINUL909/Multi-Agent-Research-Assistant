#!/usr/bin/env python3
"""
Multi-Agent Research Assistant - Fixed All-in-One Version
Real web search + AI APIs + PDF analysis in one simple file.
"""

import asyncio
import json
import logging
import uuid
import re
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SimpleConfig:
    """Simple configuration."""

    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.use_ai_summarization = bool(self.openai_api_key)
        self.max_web_results = 8
        self.timeout = 30


class WebAgent:
    """Web search agent with real API integration."""

    def __init__(self, config):
        self.config = config

    async def search(self, query: str, max_results: int = 8) -> List[Dict]:
        """Search the web using multiple methods."""
        logger.info(f"🔍 Searching web for: {query}")

        # Create session properly
        import aiohttp
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            # Try real web search first
            real_results = await self._search_real_web(session, query, max_results)

            # If real search fails, use mock data as fallback
            if not real_results:
                logger.info("Using mock data as fallback")
                real_results = await self._search_mock(query, max_results)

            # Extract content from URLs
            enriched_results = []
            for result in real_results:
                try:
                    content = await self._extract_web_content(session, result['url'])
                    if content:
                        result['content'] = content
                        result['word_count'] = len(content.split())
                    enriched_results.append(result)

                    # Small delay to be respectful
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Failed to extract content from {result['url']}: {str(e)}")
                    enriched_results.append(result)

            return enriched_results

    async def _search_real_web(self, session, query: str, max_results: int) -> List[Dict]:
        """Search real websites using APIs and web scraping."""
        results = []

        try:
            # Method 1: Try DuckDuckGo search (no API key needed)
            ddg_results = await self._search_duckduckgo(session, query, max_results // 2)
            results.extend(ddg_results)

            # Method 2: Try Wikipedia search
            wiki_results = await self._search_wikipedia(session, query, 2)
            results.extend(wiki_results)

            # Method 3: Try arXiv for academic papers (if query seems academic)
            if any(word in query.lower() for word in
                   ['research', 'study', 'analysis', 'algorithm', 'machine learning', 'ai', 'data']):
                arxiv_results = await self._search_arxiv(session, query, 2)
                results.extend(arxiv_results)

            # Method 4: Try Reddit search for discussions
            reddit_results = await self._search_reddit(session, query, 2)
            results.extend(reddit_results)

            logger.info(f"Found {len(results)} real web results")
            return results[:max_results]

        except Exception as e:
            logger.error(f"Real web search failed: {str(e)}")
            return []

    async def _search_duckduckgo(self, session, query: str, max_results: int) -> List[Dict]:
        """Search using DuckDuckGo instant answers API."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    results = []

                    # Get abstract if available
                    if data.get('Abstract'):
                        results.append({
                            'title': data.get('Heading', query),
                            'url': data.get('AbstractURL', 'https://duckduckgo.com'),
                            'content': data.get('Abstract', ''),
                            'source': 'DuckDuckGo',
                            'relevance': 0.9
                        })

                    # Get related topics
                    for topic in data.get('RelatedTopics', [])[:max_results - 1]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            results.append({
                                'title': topic.get('Text', '')[:100],
                                'url': topic.get('FirstURL', 'https://duckduckgo.com'),
                                'content': topic.get('Text', ''),
                                'source': 'DuckDuckGo',
                                'relevance': 0.7
                            })

                    return results

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {str(e)}")

        return []

    async def _search_wikipedia(self, session, query: str, max_results: int) -> List[Dict]:
        """Search Wikipedia for relevant articles."""
        try:
            # Search for articles
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"

            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get('extract'):
                        return [{
                            'title': data.get('title', query),
                            'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            'content': data.get('extract', ''),
                            'source': 'Wikipedia',
                            'relevance': 0.85
                        }]

        except Exception as e:
            logger.warning(f"Wikipedia search failed: {str(e)}")

        return []

    async def _search_arxiv(self, session, query: str, max_results: int) -> List[Dict]:
        """Search arXiv for academic papers."""
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    content = await response.text()

                    # Parse XML response (simplified)
                    results = []
                    if '<entry>' in content:
                        # Extract basic info (this is a simplified parser)
                        titles = re.findall(r'<title>(.*?)</title>', content)
                        summaries = re.findall(r'<summary>(.*?)</summary>', content, re.DOTALL)
                        links = re.findall(r'<id>(http://arxiv.org/abs/[^<]+)</id>', content)

                        for i in range(min(len(titles), len(summaries), len(links), max_results)):
                            if i > 0:  # Skip the first title which is usually the feed title
                                results.append({
                                    'title': titles[i][:200],
                                    'url': links[i - 1] if i - 1 < len(links) else '',
                                    'content': summaries[i - 1][:500] if i - 1 < len(summaries) else '',
                                    'source': 'arXiv',
                                    'relevance': 0.88
                                })

                    return results

        except Exception as e:
            logger.warning(f"arXiv search failed: {str(e)}")

        return []

    async def _search_reddit(self, session, query: str, max_results: int) -> List[Dict]:
        """Search Reddit for discussions."""
        try:
            url = f"https://www.reddit.com/search.json"
            params = {
                'q': query,
                'limit': max_results,
                'sort': 'relevance'
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    results = []
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        if post_data.get('selftext'):
                            results.append({
                                'title': post_data.get('title', '')[:200],
                                'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                'content': post_data.get('selftext', '')[:500],
                                'source': 'Reddit',
                                'relevance': 0.6
                            })

                    return results

        except Exception as e:
            logger.warning(f"Reddit search failed: {str(e)}")

        return []

    async def _extract_web_content(self, session, url: str) -> str:
        """Extract content from a web page."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()

                    # Parse HTML and extract text
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "header", "footer"]):
                        script.decompose()

                    # Get text content
                    text = soup.get_text()

                    # Clean up text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)

                    return text[:2000]  # Limit content length

        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {str(e)}")

        return ""

    async def _search_mock(self, query: str, max_results: int) -> List[Dict]:
        """Fallback mock search results."""
        results = [
            {
                'title': f'Research Study: {query}',
                'url': 'https://research-example.com/study',
                'content': f'Comprehensive research analysis of {query}. This study examines current trends, methodologies, and applications. Key findings include significant developments in the field with practical implications for implementation.',
                'word_count': 250,
                'relevance': 0.95,
                'source': 'Mock'
            },
            {
                'title': f'{query} - Industry Report 2024',
                'url': 'https://industry-report.com/2024',
                'content': f'Latest industry analysis covering {query} market trends, growth projections, and key players. The report highlights emerging opportunities and challenges facing the sector.',
                'word_count': 200,
                'relevance': 0.88,
                'source': 'Mock'
            },
            {
                'title': f'Complete Guide to {query}',
                'url': 'https://guide-site.com/complete',
                'content': f'Comprehensive guide covering fundamentals and advanced concepts of {query}. Includes practical examples, best practices, and implementation strategies for professionals.',
                'word_count': 180,
                'relevance': 0.82,
                'source': 'Mock'
            }
        ]

        return results[:max_results]


class PDFAgent:
    """Simple PDF processing agent."""

    def __init__(self, config):
        self.config = config

    async def extract(self, pdf_path: str) -> Dict:
        """Extract content from PDF."""
        logger.info(f"📄 Processing PDF: {pdf_path}")

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return self._create_error_result(pdf_path, "File not found")

        # Try PyMuPDF first
        try:
            import fitz
            return await self._extract_with_pymupdf(pdf_path)
        except ImportError:
            pass

        # Try PyPDF2
        try:
            import PyPDF2
            return await self._extract_with_pypdf2(pdf_path)
        except ImportError:
            pass

        return self._create_error_result(pdf_path, "No PDF libraries available")

    async def _extract_with_pymupdf(self, pdf_path: str):
        """Extract using PyMuPDF."""
        import fitz

        doc = fitz.open(pdf_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()

        doc.close()

        return self._process_content(text, Path(pdf_path).name, len(doc))

    async def _extract_with_pypdf2(self, pdf_path: str):
        """Extract using PyPDF2."""
        import PyPDF2

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""

            for page in reader.pages:
                text += page.extract_text()

        return self._process_content(text, Path(pdf_path).name, len(reader.pages))

    def _process_content(self, text: str, filename: str, page_count: int):
        """Process extracted text."""
        # Clean text
        clean_text = re.sub(r'\s+', ' ', text).strip()

        # Extract key points
        key_points = []
        for line in text.split('\n'):
            line = line.strip()
            if (line.startswith('•') or line.startswith('-') or
                re.match(r'^\d+\.', line)) and len(line) > 10:
                key_points.append(line[:200])

        return {
            'filename': filename,
            'text': clean_text,
            'page_count': page_count,
            'key_points': key_points[:10],
            'word_count': len(clean_text.split()),
            'success': True
        }

    def _create_error_result(self, pdf_path: str, error: str):
        """Create error result."""
        return {
            'filename': Path(pdf_path).name,
            'text': '',
            'page_count': 0,
            'key_points': [],
            'word_count': 0,
            'success': False,
            'error': error
        }


class SummaryAgent:
    """Summarization agent with AI API integration."""

    def __init__(self, config):
        self.config = config
        self.openai_available = False

        # Check for OpenAI API
        if config.openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
                self.openai_available = True
                logger.info("🤖 OpenAI API available for AI summarization")
            except ImportError:
                logger.info("🔤 OpenAI not installed, using extractive summarization")
        else:
            logger.info("🔤 Using extractive summarization (no OpenAI key)")

    async def summarize(self, data: Dict) -> str:
        """Create summary using AI API or extractive methods."""
        query = data.get('query', '')
        web_results = data.get('web_results', [])
        pdf_results = data.get('pdf_results', [])

        logger.info(f"📝 Creating summary for {len(web_results)} web + {len(pdf_results)} PDF sources")

        # Try AI summarization first
        if self.openai_available and self.config.use_ai_summarization:
            try:
                return await self._ai_summarize(query, web_results, pdf_results)
            except Exception as e:
                logger.warning(f"AI summarization failed: {str(e)}, using extractive method")

        # Fallback to extractive summarization
        return await self._extractive_summarize(query, web_results, pdf_results)

    async def _ai_summarize(self, query: str, web_results: List[Dict], pdf_results: List[Dict]) -> str:
        """Use OpenAI API for intelligent summarization."""

        # Prepare content for AI
        content_parts = [f"Research Query: {query}\n"]

        # Add web sources
        if web_results:
            content_parts.append("WEB SOURCES:")
            for i, result in enumerate(web_results[:5], 1):
                content = result.get('content', '')[:400]  # Limit for API
                source = result.get('source', 'Web')
                content_parts.append(f"{i}. [{source}] {result['title']}")
                content_parts.append(f"   {content}...")
                content_parts.append("")

        # Add PDF sources
        if pdf_results:
            content_parts.append("PDF SOURCES:")
            for i, pdf in enumerate(pdf_results, 1):
                if pdf.get('success'):
                    text_preview = pdf.get('text', '')[:600]
                    content_parts.append(f"{i}. {pdf['filename']}")
                    content_parts.append(f"   {text_preview}...")
                    if pdf.get('key_points'):
                        content_parts.append(f"   Key Points: {'; '.join(pdf['key_points'][:3])}")
                    content_parts.append("")

        prompt = "\n".join(content_parts)
        prompt += "\n\nPlease provide a comprehensive research summary that includes:\n1. Executive summary (2-3 sentences)\n2. Key findings from sources\n3. Main themes and trends\n4. Practical implications\n5. Conclusion with actionable insights"

        try:
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are an expert research analyst. Provide clear, comprehensive, and well-structured summaries that synthesize information from multiple sources."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )

            ai_summary = response.choices[0].message.content

            # Add metadata
            summary_with_meta = f"# AI-Generated Research Summary: {query}\n\n"
            summary_with_meta += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            summary_with_meta += f"**Sources:** {len(web_results)} web, {len(pdf_results)} PDF\n"
            summary_with_meta += f"**Method:** OpenAI GPT Analysis\n\n"
            summary_with_meta += ai_summary

            return summary_with_meta

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    async def _extractive_summarize(self, query: str, web_results: List[Dict], pdf_results: List[Dict]) -> str:
        """Create summary using extractive methods."""

        summary_parts = [
            f"# Research Summary: {query}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Sources:** {len(web_results)} web, {len(pdf_results)} PDF",
            f"**Method:** Extractive Analysis\n"
        ]

        # Analyze themes
        themes = self._extract_themes(web_results, pdf_results, query)
        if themes:
            summary_parts.append("## Key Themes")
            for i, (theme, count) in enumerate(themes[:5], 1):
                summary_parts.append(f"{i}. **{theme.title()}** (mentioned {count} times)")
            summary_parts.append("")

        # Key findings from web
        if web_results:
            summary_parts.append("## Web Research Findings")
            key_sentences = self._extract_key_sentences(web_results, query)
            for i, sentence in enumerate(key_sentences[:4], 1):
                summary_parts.append(f"**{i}.** {sentence['text']}")
                summary_parts.append(f"   *Source: {sentence['source']}*")
            summary_parts.append("")

        # Key findings from PDFs
        if pdf_results:
            summary_parts.append("## PDF Document Analysis")
            for pdf in pdf_results:
                if pdf.get('success'):
                    summary_parts.append(f"### {pdf['filename']}")
                    summary_parts.append(f"- **Pages:** {pdf['page_count']}")
                    summary_parts.append(f"- **Word Count:** {pdf['word_count']}")
                    if pdf.get('key_points'):
                        summary_parts.append("- **Key Points:**")
                        for point in pdf['key_points'][:3]:
                            summary_parts.append(f"  • {point}")
                    summary_parts.append("")

        # Generate insights
        insights = self._generate_insights(web_results, pdf_results, query)
        if insights:
            summary_parts.append("## Key Insights")
            for insight in insights:
                summary_parts.append(f"• {insight}")
            summary_parts.append("")

        # Statistics
        total_sources = len(web_results) + len([p for p in pdf_results if p.get('success')])
        total_words = sum(r.get('word_count', 0) for r in web_results)
        total_words += sum(p.get('word_count', 0) for p in pdf_results if p.get('success'))

        summary_parts.extend([
            "## Research Statistics",
            f"- **Total Sources:** {total_sources}",
            f"- **Web Sources:** {len(web_results)}",
            f"- **PDF Sources:** {len([p for p in pdf_results if p.get('success')])}",
            f"- **Total Content:** {total_words:,} words",
            f"- **Real Web Data:** {len([r for r in web_results if r.get('source', '') != 'Mock'])} sources"
        ])

        return "\n".join(summary_parts)

    def _extract_themes(self, web_results: List[Dict], pdf_results: List[Dict], query: str) -> List[tuple]:
        """Extract key themes from content."""
        all_text = query + " "

        # Collect text from all sources
        for result in web_results:
            all_text += result.get('content', '') + " " + result.get('title', '') + " "

        for pdf in pdf_results:
            if pdf.get('success'):
                all_text += pdf.get('text', '')[:1000] + " "
                all_text += " ".join(pdf.get('key_points', [])) + " "

        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())

        # Filter stop words
        stop_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were',
            'said', 'each', 'which', 'their', 'time', 'about', 'would', 'there',
            'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first',
            'research', 'study', 'analysis', 'data', 'results', 'content', 'source',
            'using', 'used', 'based', 'show', 'find', 'found', 'include', 'such'
        }

        filtered_words = [w for w in words if w not in stop_words and len(w) > 4]
        word_counts = Counter(filtered_words)

        return word_counts.most_common(8)

    def _extract_key_sentences(self, web_results: List[Dict], query: str) -> List[Dict]:
        """Extract most relevant sentences."""
        key_sentences = []
        query_terms = set(query.lower().split())

        for result in web_results:
            content = result.get('content', '')
            title = result.get('title', 'Unknown')
            source = result.get('source', 'Web')

            if not content:
                continue

            sentences = re.split(r'[.!?]+', content)

            for sentence in sentences:
                sentence = sentence.strip()
                if 20 <= len(sentence) <= 200:  # Good length sentences
                    # Calculate relevance
                    sentence_words = set(sentence.lower().split())
                    relevance = len(query_terms.intersection(sentence_words)) / max(len(query_terms), 1)

                    if relevance > 0.1:
                        key_sentences.append({
                            'text': sentence,
                            'source': f"{source} - {title[:30]}...",
                            'relevance': relevance
                        })

        # Sort by relevance
        key_sentences.sort(key=lambda x: x['relevance'], reverse=True)
        return key_sentences[:6]

    def _generate_insights(self, web_results: List[Dict], pdf_results: List[Dict], query: str) -> List[str]:
        """Generate insights from the research."""
        insights = []

        total_sources = len(web_results) + len([p for p in pdf_results if p.get('success')])
        real_web_sources = len([r for r in web_results if r.get('source', '') != 'Mock'])

        if total_sources > 0:
            insights.append(f"Analysis covers {total_sources} diverse sources including web articles and documents")

        if real_web_sources > 0:
            insights.append(f"Includes {real_web_sources} real-time web sources with current information")

        # Source diversity
        sources = set(r.get('source', 'Unknown') for r in web_results)
        if len(sources) > 2:
            insights.append(
                f"Information gathered from {len(sources)} different platforms: {', '.join(list(sources)[:3])}")

        # Academic content
        academic_sources = len([p for p in pdf_results if p.get('success')])
        if academic_sources > 0:
            insights.append(f"In-depth analysis supported by {academic_sources} PDF documents")

        # Content volume
        total_words = sum(r.get('word_count', 0) for r in web_results)
        total_words += sum(p.get('word_count', 0) for p in pdf_results if p.get('success'))

        if total_words > 1000:
            insights.append(f"Comprehensive analysis of {total_words:,} words of research content")

        return insights[:5]


class ReportAgent:
    """Simple report generation agent."""

    def __init__(self, config):
        self.config = config

    async def generate(self, data: Dict) -> str:
        """Generate research report."""
        query = data.get('query', '')
        summary = data.get('summary', '')
        web_results = data.get('web_results', [])
        pdf_results = data.get('pdf_results', [])

        logger.info(f"📊 Generating report for: {query}")

        report_parts = [
            f"# Research Report: {query}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Sources:** {len(web_results)} web, {len(pdf_results)} PDF\n",

            "## Executive Summary",
            summary,
            "\n## Detailed Sources\n"
        ]

        # Web sources
        if web_results:
            report_parts.append("### Web Sources")
            for i, result in enumerate(web_results, 1):
                source_info = f"[{result.get('source', 'Web')}]"
                report_parts.extend([
                    f"**{i}. {result['title']} {source_info}**",
                    f"URL: {result['url']}",
                    f"Content: {result.get('content', '')[:300]}...\n"
                ])

        # PDF sources
        if pdf_results:
            report_parts.append("### PDF Sources")
            for i, pdf in enumerate(pdf_results, 1):
                if pdf.get('success'):
                    report_parts.extend([
                        f"**{i}. {pdf['filename']}**",
                        f"Pages: {pdf['page_count']}, Words: {pdf['word_count']}",
                        f"Key Points: {'; '.join(pdf.get('key_points', [])[:2])}\n"
                    ])

        report_parts.append("---\n*Generated by Multi-Agent Research Assistant*")

        return "\n".join(report_parts)


class ResearchAssistant:
    """Main research assistant - coordinates all agents."""

    def __init__(self):
        self.config = SimpleConfig()
        self.web_agent = WebAgent(self.config)
        self.pdf_agent = PDFAgent(self.config)
        self.summary_agent = SummaryAgent(self.config)
        self.report_agent = ReportAgent(self.config)

        # Create output directory
        Path("outputs").mkdir(exist_ok=True)

    async def research(
            self,
            query: str,
            pdf_files: List[str] = None,
            max_web_results: int = 8
    ) -> Dict:
        """Conduct complete research."""
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"🚀 Starting research task {task_id}: {query}")

        try:
            # 1. Web Search
            web_results = await self.web_agent.search(query, max_web_results)

            # 2. PDF Processing
            pdf_results = []
            if pdf_files:
                for pdf_file in pdf_files:
                    result = await self.pdf_agent.extract(pdf_file)
                    pdf_results.append(result)

            # 3. Summarization
            summary = await self.summary_agent.summarize({
                'query': query,
                'web_results': web_results,
                'pdf_results': pdf_results
            })

            # 4. Report Generation
            report = await self.report_agent.generate({
                'query': query,
                'summary': summary,
                'web_results': web_results,
                'pdf_results': pdf_results
            })

            # 5. Save Results
            results = {
                'task_id': task_id,
                'query': query,
                'web_results': web_results,
                'pdf_results': pdf_results,
                'summary': summary,
                'report': report,
                'created_at': datetime.now().isoformat()
            }

            await self._save_results(task_id, results)

            logger.info(f"✅ Research completed: {task_id}")
            return results

        except Exception as e:
            logger.error(f"❌ Research failed: {str(e)}")
            raise

    async def _save_results(self, task_id: str, results: Dict):
        """Save results to files."""
        task_dir = Path(f"outputs/task_{task_id}")
        task_dir.mkdir(exist_ok=True)

        # Save complete results
        with open(task_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        # Save report
        with open(task_dir / "report.md", 'w', encoding='utf-8') as f:
            f.write(results['report'])

        # Save summary
        with open(task_dir / "summary.md", 'w', encoding='utf-8') as f:
            f.write(results['summary'])

        logger.info(f"💾 Results saved to: {task_dir}")


# Main execution
async def main():
    """Example usage."""
    print("🔬 Multi-Agent Research Assistant")
    print("=" * 40)

    # Initialize assistant
    assistant = ResearchAssistant()

    # Example research
    results = await assistant.research(
        query="artificial intelligence ",
        pdf_files=[],  # Add PDF paths here if you have them
        max_web_results=5
    )

    print(f"\n✅ Research Complete!")
    print(f"📁 Task ID: {results['task_id']}")
    print(f"🌐 Web sources: {len(results['web_results'])}")
    print(f"📄 PDF sources: {len(results['pdf_results'])}")
    print(f"📂 Check: outputs/task_{results['task_id']}/")

    # Show summary preview
    summary = results['summary']
    print(f"\n📝 Summary Preview:")
    print("-" * 30)
    print(summary[:400] + "..." if len(summary) > 400 else summary)


if __name__ == "__main__":
    asyncio.run(main())
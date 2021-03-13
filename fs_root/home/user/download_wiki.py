import wikipediaapi
import pickle

def fetch_category(category="Tom Cruise filmography"):
    # Get all links in a wiki page
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(category)
    def get_links(page):
        links = page.links
        for title in sorted(links.keys()):
            if links[title].ns != 0 or "Unauthorized" in title:
                links.pop(title)
        return links

    links = get_links(page_py)

    pages_text = {}
    for title in links:
        page_py = links[title]
        pages_text[title] = page_py.text

    pages_text

    for title in pages_text:
        stop_index = pages_text[title].rfind("References")
        pages_text[title] = pages_text[title][:stop_index]

    return pages_text

d = fetch_category()
with open('pages.pkl', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

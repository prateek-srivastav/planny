"""Traveller's country -> ISO code and currency. Small, static and offline on purpose.

A traveller can be from anywhere. This table is only used to *default* the budget currency and to
localise search results to their market; entry rules and everything else come from research, so an
unknown country still works (the text is passed to the research agent as written, nothing is
localised, currency falls back to USD).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Country:
    name: str
    iso2: str
    currency: str


_ROWS = """
India|IN|INR;United States|US|USD;United Kingdom|GB|GBP;Canada|CA|CAD;Australia|AU|AUD
New Zealand|NZ|NZD;Germany|DE|EUR;France|FR|EUR;Italy|IT|EUR;Spain|ES|EUR;Netherlands|NL|EUR
Ireland|IE|EUR;Portugal|PT|EUR;Belgium|BE|EUR;Austria|AT|EUR;Finland|FI|EUR;Greece|GR|EUR
Switzerland|CH|CHF;Sweden|SE|SEK;Norway|NO|NOK;Denmark|DK|DKK;Poland|PL|PLN;Czechia|CZ|CZK
Hungary|HU|HUF;Romania|RO|RON;Turkey|TR|TRY;Ukraine|UA|UAH;Russia|RU|RUB;Israel|IL|ILS
United Arab Emirates|AE|AED;Saudi Arabia|SA|SAR;Qatar|QA|QAR;Kuwait|KW|KWD;Oman|OM|OMR
Bahrain|BH|BHD;Egypt|EG|EGP;South Africa|ZA|ZAR;Nigeria|NG|NGN;Kenya|KE|KES;Ghana|GH|GHS
Ethiopia|ET|ETB;Morocco|MA|MAD;Tanzania|TZ|TZS;Pakistan|PK|PKR;Bangladesh|BD|BDT
Sri Lanka|LK|LKR;Nepal|NP|NPR;Maldives|MV|MVR;China|CN|CNY;Hong Kong|HK|HKD;Taiwan|TW|TWD
Japan|JP|JPY;South Korea|KR|KRW;Singapore|SG|SGD;Malaysia|MY|MYR;Thailand|TH|THB
Vietnam|VN|VND;Indonesia|ID|IDR;Philippines|PH|PHP;Brazil|BR|BRL;Argentina|AR|ARS
Chile|CL|CLP;Colombia|CO|COP;Peru|PE|PEN;Mexico|MX|MXN
"""

COUNTRIES: list[Country] = [
    Country(*row.split("|")) for chunk in _ROWS.strip().splitlines() for row in chunk.split(";")
]

_ALIASES = {
    "usa": "US", "america": "US", "american": "US", "uk": "GB", "britain": "GB",
    "england": "GB", "british": "GB", "uae": "AE", "emirati": "AE", "korea": "KR",
    "korean": "KR", "holland": "NL", "dutch": "NL", "czech republic": "CZ", "turkiye": "TR",
    "indian": "IN", "canadian": "CA", "australian": "AU", "german": "DE", "french": "FR",
    "italian": "IT", "spanish": "ES", "irish": "IE", "swiss": "CH", "swedish": "SE",
    "japanese": "JP", "chinese": "CN", "singaporean": "SG", "malaysian": "MY", "thai": "TH",
    "vietnamese": "VN", "indonesian": "ID", "filipino": "PH", "brazilian": "BR",
    "mexican": "MX", "nigerian": "NG", "kenyan": "KE", "south african": "ZA",
    "pakistani": "PK", "bangladeshi": "BD", "sri lankan": "LK", "nepali": "NP",
    "egyptian": "EG", "moroccan": "MA", "turkish": "TR", "russian": "RU",
}  # fmt: skip

_BY_KEY: dict[str, Country] = {}
for _c in COUNTRIES:
    _BY_KEY[_c.name.lower()] = _c
    _BY_KEY[_c.iso2.lower()] = _c
for _alias, _iso in _ALIASES.items():
    _BY_KEY[_alias] = _BY_KEY[_iso.lower()]


def resolve(text: str | None) -> Country | None:
    """Match a country name, ISO-2 code, or common demonym ('Indian'). None if unknown."""
    if not text:
        return None
    return _BY_KEY.get(" ".join(text.lower().replace(".", "").split()))

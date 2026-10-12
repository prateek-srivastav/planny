"""Market-specific guidance, injected into the agents' prompts for travellers from that country.

Nothing here is required: a traveller from a country with no entry gets the generic prompts
(currency, home-market sources, entry rules for their nationality). Add a country by adding a key.
"""

MARKET_NOTES: dict[str, str] = {
    "IN": (
        "Market notes for travellers from India: write amounts in whole rupees when the budget is "
        "INR, and prefer Indian travel sites and blogs for costs. Consider season and monsoon "
        "timing, school and public holidays and festival crowds, how to get there (flights, "
        "trains) and around (auto-rickshaws, cab apps, metro, local trains), and vegetarian, "
        "Jain and halal food availability. For a foreign destination cover the visa or permit "
        "situation for Indian passport holders and whether INR cards or UPI work there. Price "
        "hotels per room per night, and meals at local restaurants, not tourist rates."
    ),
}


def notes_for(iso2: str | None) -> str:
    return MARKET_NOTES.get(iso2 or "", "")

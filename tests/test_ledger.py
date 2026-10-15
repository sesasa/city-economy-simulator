import pytest

from citysim.core.ledger import Ledger, LedgerError, open_standard_external_accounts


def make():
    L = Ledger()
    open_standard_external_accounts(L)
    L.open_account("a"); L.open_account("b")
    L.register_item("corn", "good")
    return L


def test_double_entry_sums_to_zero():
    L = make()
    L.produce("a", "corn", 10)
    L.transfer("a", "b", "corn", 4)
    L.remove("b", "corn", 1, "consumption")
    assert L.balance("a", "corn") == 6 and L.balance("b", "corn") == 3
    assert sum(v for (_, i), v in L.balances.items() if i == "corn") == 0
    assert L.internal_total("corn") == 9
    assert [p.flow for p in L.journal] == ["production", "transfer", "consumption"]


@pytest.mark.parametrize("call", [
    lambda L: L.post("a", "b", "corn", -1, "transfer"),            # negative amount
    lambda L: L.post("a", "b", "corn", 1, "production"),           # production must credit a source
    lambda L: L.post("a", "ext:consumption", "corn", 1, "transfer"),  # transfer to external
    lambda L: L.post("a", "b", "gold", 1, "transfer"),             # unregistered item
    lambda L: L.post("a", "b", "corn", 1, "magic"),                # unknown flow
    lambda L: L.post("a", "zzz", "corn", 1, "transfer"),           # unknown account
])
def test_rejects_malformed_postings(call):
    with pytest.raises(LedgerError):
        call(make())


def test_zero_posting_is_noop():
    L = make()
    L.transfer("a", "b", "corn", 0.0)
    assert L.journal == []

import unittest

from bbg import BloombergClient, BloombergConnection


class _FakeRequestList:
    def __init__(self) -> None:
        self.values = []

    def appendValue(self, value) -> None:
        self.values.append(value)


class _FakeRequest:
    def __init__(self) -> None:
        self._elements = {
            "securities": _FakeRequestList(),
            "fields": _FakeRequestList(),
        }
        self.settings = {}

    def append(self, name: str, value) -> None:
        self._elements.setdefault(name, _FakeRequestList()).appendValue(value)

    def getElement(self, name: str):
        self._elements.setdefault(name, _FakeRequestList())
        return self._elements[name]

    def set(self, name: str, value) -> None:
        self.settings[name] = value


class _FakeService:
    def createRequest(self, _request_type: str) -> _FakeRequest:
        return _FakeRequest()


class _FakeElement:
    def __init__(self, mapping=None, values=None) -> None:
        self.mapping = mapping or {}
        self.values = values

    def asElement(self):
        return self

    def hasElement(self, name: str) -> bool:
        return name in self.mapping

    def getElement(self, name: str):
        return self.mapping[name]

    def getElementAsString(self, name: str) -> str:
        return str(self.mapping[name])

    def numValues(self) -> int:
        return len(self.values or [])

    def getValueAsElement(self, idx: int):
        return (self.values or [])[idx]

    def getValue(self, idx: int):
        return (self.values or [])[idx]


class _FakeMessage:
    def __init__(self, message_type: str, element: _FakeElement) -> None:
        self._message_type = message_type
        self._element = element

    def messageType(self) -> str:
        return self._message_type

    def asElement(self) -> _FakeElement:
        return self._element


class _FakeEvent:
    def __init__(self, messages, event_type: int) -> None:
        self._messages = list(messages)
        self._event_type = event_type

    def __iter__(self):
        return iter(self._messages)

    def eventType(self) -> int:
        return self._event_type


class _FakeSession:
    def __init__(self, events) -> None:
        self._events = list(events)
        self.requests = []

    def sendRequest(self, req) -> None:
        self.requests.append(req)

    def nextEvent(self):
        return self._events.pop(0)


class _FakeBlpapi:
    class Event:
        RESPONSE = 1


def _make_client(events) -> BloombergClient:
    client = object.__new__(BloombergClient)
    client.conn = BloombergConnection()
    client._blpapi = _FakeBlpapi
    client._session = _FakeSession(events)
    client._service = _FakeService()
    return client


class BloombergResponseParsingTests(unittest.TestCase):
    def test_get_active_chain_parses_reference_security_data(self) -> None:
        chain_item = _FakeElement(mapping={"Security Description": "COZ6 Comdty"})
        chain = _FakeElement(values=[chain_item])
        field_data = _FakeElement(mapping={"FUT_CHAIN": chain})
        sec_item = _FakeElement(mapping={"fieldData": field_data})
        security_data = _FakeElement(values=[sec_item])
        msg = _FakeMessage(
            "ReferenceDataResponse",
            _FakeElement(mapping={"securityData": security_data}),
        )
        event = _FakeEvent([msg], _FakeBlpapi.Event.RESPONSE)

        client = _make_client([event])

        self.assertEqual(client.get_active_chain("CO"), {"COZ6 COMDTY"})

    def test_get_active_chain_skips_reference_response_without_security_data(self) -> None:
        msg = _FakeMessage(
            "ReferenceDataResponse",
            _FakeElement(mapping={"responseError": _FakeElement(mapping={"message": "bad request"})}),
        )
        event = _FakeEvent([msg], _FakeBlpapi.Event.RESPONSE)

        client = _make_client([event])

        self.assertEqual(client.get_active_chain("CO"), set())

    def test_get_snapshot_skips_reference_response_without_security_data(self) -> None:
        msg = _FakeMessage(
            "ReferenceDataResponse",
            _FakeElement(mapping={"responseError": _FakeElement(mapping={"message": "bad request"})}),
        )
        event = _FakeEvent([msg], _FakeBlpapi.Event.RESPONSE)

        client = _make_client([event])
        out = client.get_snapshot(["COZ6 Comdty"])

        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()

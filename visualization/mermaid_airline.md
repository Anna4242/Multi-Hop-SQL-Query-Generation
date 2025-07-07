# AIRLINE Database Schema

```mermaid
graph TD
    Air Carriers["Air Carriers<br/>Code<br/>Description"]
    Airports["Airports<br/>Code<br/>Description"]
    Airlines["Airlines<br/>FL_DATE<br/>OP_CARRIER_AIRLINE_ID<br/>TAIL_NUM<br/>... +25 more"]
    Airlines --> Airports
    Airports --> Airlines
    Air Carriers --> Airlines
    Airports --> Air Carriers
    Air Carriers --> Airports
    Airlines --> Air Carriers
```

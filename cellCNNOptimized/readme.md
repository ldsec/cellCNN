# CellCNN Optimized

## Parameters

The ML and crypto parameters can be set throuh the file `params.go`.

The following inequality must be respected to ensure the ciphertext has enough available slots to store all the data :

_3onh+(2o+1)(nh+(m/2−1)2h+h)≤N/2_

for _o_ the number of `Classes` (labels), _h_ the number of `Features`, _m_ the number of `Filters` (markers) and _n_ the `BatchSize`. In this implementation, the number of `Cells` per `Sample` does not impact the slot usage or training time.

The value `LogN` can be increase to accomodate for more slots.

The method `GenParams()` will automatically generate a `ckks.Parameters` with secure parameters for the circuit. 

## Testing

The file `decentralized/example/main.go` is an example of decentralized training (the centralized training can be emulated by setting the number of hosts to 1).


> Fix rel-int/discopy-next#252 by taking the default of our ask there (comment 6405499692, asked 10-26, recommending (A)), now past its 48-hour due date with no word from USER: `channel.Sum.double` doubles a sum coherently, keeping the cross terms, so that it agrees with `get_kraus`, which reads the same sum as coherent. The test `(BS + BS).double() == 2 * BS.double()` becomes 4×.

- [x] `channel.Sum.double` doubles coherently, the doubling of the channel whose Kraus map is `get_kraus()`; an impure term refuses
- [x] tests: the Bell sum keeps its cross terms and agrees with the density matrix of `get_kraus`; `(BS + BS).double()` is 4x
- [ ] CHANGELOG bullet

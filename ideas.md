# The Doodler

## Modes

- AI face off
  - you draw something, send link to friend, friend picks the better one (between yours and one that AI drew)
- Turing test
  - you draw something, send link to friend, they have to guess which one was yours (between yours and one that AI drew)
- normal face off
  - You doodle, send link to friend, they draw something, ai chooses better doodle by how fast it could recognize
- Solo perfection mode
  - You draw something, ai ranks it by how fast it could recognize it
  - Time gets put on leaderboard

What resolution? Maybe 256 x 256 (multiple of some power of 2 would be good)
Call it R for now (RxR image)

## Models

### Doodler Model (doodle category -> stroke array)

- Trained on google’s draw it! Database
- Either a model for each category or one big model that takes in the doodle category plaintext
- Use stroke order and time things
- Also get the image afterwards using the stroke array

### Classifier Model (RxR image -> confidence vector on all classifications)

- Trained on google’s draw it! Database
- Probably some kind of CNN

### Judger Model (RxR image -> number score)

- Could use the Drawer model and see variance from it but would be bad
- See if we have any way of having the doodles evaluated
- Idea: Maybe check if we can just use the Classifier model and see how fast it can predict what it is? (Score is time)

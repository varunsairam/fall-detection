/* The classifier was made with the help of MicroML (https://github.com/eloquentarduino/micromlgen) 
where the Logistic Regression model from Python and sklearn was converted into C friendly code */

#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class LogisticRegression {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float votes[2] = { 0.0f };
                        votes[0] = dot(x,   -1.060813830246  , -0.732475827801  , 1.642059310698  , 2.969936271294  , 1.456510933796  , -0.294101941638  , 0.786678492554  , 2.622968189837 );
                        // return argmax of votes
                        uint8_t classIdx = 0;
                        float maxVotes = votes[0];

                        for (uint8_t i = 1; i < 2; i++) {
                            if (votes[i] < maxVotes) { // Changed sign direction
                                classIdx = i;
                                maxVotes = votes[i];
                            }
                        }
                        return classIdx;
                    }

                protected:
                    /**
                    * Compute dot product
                    */
                    float dot(float *x, ...) {
                        va_list w;
                        va_start(w, 8);
                        float dot = 0.0;

                        for (uint16_t i = 0; i < 8; i++) {
                            const float wi = va_arg(w, double);
                            dot += x[i] * wi;
                        }

                        return dot;
                    }
                };
            }
        }
    }
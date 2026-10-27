// Gradient-Check: Autograd-Gradienten müssen mit numerischen Gradienten
// (f(x+h) − f(x−h)) / 2h übereinstimmen.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Value } from '../public/js/value.js';
import { softmax, crossEntropy } from '../public/js/nn.js';

function numericGrad(f, xs, i, h = 1e-6) {
  const plus = xs.slice(); plus[i] += h;
  const minus = xs.slice(); minus[i] -= h;
  return (f(plus.map((x) => new Value(x))).data - f(minus.map((x) => new Value(x))).data) / (2 * h);
}

function check(name, f, xs) {
  test(name, () => {
    const vs = xs.map((x) => new Value(x));
    f(vs).backward();
    xs.forEach((_, i) => {
      const expected = numericGrad(f, xs, i);
      assert.ok(Math.abs(vs[i].grad - expected) < 1e-6, `${name} x${i}: ${vs[i].grad} vs ${expected}`);
    });
  });
}

check('add', ([a, b]) => a.add(b), [1.5, -2]);
check('mul', ([a, b]) => a.mul(b), [1.5, -2]);
check('pow', ([a]) => a.pow(3), [1.3]);
check('pow negativ', ([a]) => a.pow(-0.5), [2.2]);
check('exp', ([a]) => a.exp(), [0.7]);
check('log', ([a]) => a.log(), [2.5]);
check('relu positiv', ([a]) => a.relu(), [0.8]);
check('relu negativ', ([a]) => a.relu(), [-0.8]);
check('tanh', ([a]) => a.tanh(), [0.4]);
check('sub/div', ([a, b]) => a.sub(b).div(b), [3, 1.7]);
check('sum', (xs) => Value.sum(xs), [1, 2, 3]);
check('mehrfach benutzt (a·a + a)', ([a]) => a.mul(a).add(a), [1.7]);
check('neuron', ([x1, x2, w1, w2, b]) => x1.mul(w1).add(x2.mul(w2)).add(b).tanh(), [2, 0, -3, 1, 6.8813735870195432]);
check('softmax[1]', (xs) => softmax(xs)[1], [0.2, 1.5, -0.3]);
check('crossEntropy', (xs) => crossEntropy(xs, 2), [0.2, 1.5, -0.3, 2.0]);

test('micrograd-Klassiker: o = 0.7071, dx1 = -1.5', () => {
  const x1 = new Value(2), x2 = new Value(0), w1 = new Value(-3), w2 = new Value(1), b = new Value(6.8813735870195432);
  const o = x1.mul(w1).add(x2.mul(w2)).add(b).tanh();
  o.backward();
  assert.ok(Math.abs(o.data - Math.SQRT1_2) < 1e-6);
  assert.ok(Math.abs(x1.grad - -1.5) < 1e-6);
  assert.ok(Math.abs(w1.grad - 1) < 1e-6);
});

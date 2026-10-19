import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/reviewData.ts', import.meta.url), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { correctionPatch, displayValue, parseAmount, equalEvidenceValue } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));

test('absent values stay blank, real zero stays zero', () => {
  assert.equal(displayValue(null), ''); assert.equal(displayValue(undefined), ''); assert.equal(displayValue(0), '0');
  assert.equal(parseAmount(''), null); assert.equal(parseAmount('0'), 0);
});
test('French printed amounts keep their precision', () => {
  assert.equal(parseAmount('2 640,753'), 2640.753);
  assert.equal(parseAmount('0,831'), 0.831);
});
test('invalid monetary input cannot silently become zero', () => {
  for (const value of ['NaN', 'Infinity', '19%', '1,2,3', '12 EUR']) assert.throws(() => parseAmount(value));
});
test('unchanged nulls and unknown fields are not sent', () => {
  const original = { subtotal: null, currency: 'TND', additional_data: { workflow_default: true }, taxes: [{ label: 'FODEC', amount: 50.45 }] };
  assert.deepEqual(correctionPatch(original, { ...original }), {});
});
test('partial human correction preserves extraction metadata and taxes', () => {
  const original = { vendor_name: 'OCR NAME', taxes: [{ label: 'TVA', amount: 5.5 }], items: [{ unit: 'm3', reference: 'REF/1' }] };
  assert.deepEqual(correctionPatch(original, { ...original, vendor_name: 'Verified name' }), { vendor_name: 'Verified name' });
});
test('tax identifier aliases stay synchronized without stripping separators', () => {
  assert.deepEqual(correctionPatch({ vendor_tax_id: '123' }, { vendor_tax_id: ' 123/A-M/000 ' }), { vendor_tax_id: '123/A-M/000', tax_id: '123/A-M/000' });
});
test('cleared numeric fields are null and not computed', () => {
  assert.deepEqual(correctionPatch({ subtotal: 12 }, { subtotal: '' }), { subtotal: null });
});
test('edited article preserves unit reference source and nullable amounts', () => {
  const item = { description: 'Original', unit: 'm3', reference: 'A-2', source: 'L17', quantity: null, unit_price: 6, total: null };
  const result = correctionPatch({ items: [item] }, { items: [{ ...item, description: 'Corrected', unit_price: '6,500' }] });
  assert.deepEqual(result.items, [{ ...item, description: 'Corrected', unit_price: 6.5 }]);
});
test('explicit removal of all articles is retained', () => {
  assert.deepEqual(correctionPatch({ items: [{ description: 'Product' }] }, { items: [] }), { items: [] });
});
test('evidence accepts amount formatting but not a changed value', () => {
  assert.equal(equalEvidenceValue('12,600', 12.6), true);
  assert.equal(equalEvidenceValue('13,600', 12.6), false);
});
test('identifier truncation is not presented as the same evidence', () => {
  assert.equal(equalEvidenceValue('1362434', '1362434/S/P/M/000'), false);
  assert.equal(equalEvidenceValue('1362434 /S/P/M/000', '1362434/S/P/M/000'), true);
});

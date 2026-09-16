const assert = require('node:assert/strict');
const {computeEpisodeStats} = require('../offline/episode_stats.js');

const scene = {
  frame_count: 3,
  frames: [
    {frame_index: 0, instances: [
      {class_name: 'chair', confidence: 0.90},
      {class_name: 'chair', confidence: 0.80},
      {class_name: 'table', confidence: 0.40},
    ]},
    {frame_index: 10, instances: [
      {class_name: 'chair', confidence: 0.70},
      {class_name: 'table', confidence: 0.95},
    ]},
    {frame_index: 20, instances: [{class_name: 'table', confidence: 0.60}]},
  ],
};

const stats = computeEpisodeStats(scene, 0.50, 1);
assert.equal(stats.detected_frames, 3);
assert.equal(stats.class_count, 2);
assert.deepEqual(stats.classes.map(item => item.class_name), ['chair', 'table']);
assert.equal(stats.classes[0].detected_frames, 2);
assert.equal(stats.classes[0].instance_count, 3);
assert.equal(stats.classes[0].representative_frame, 0);
assert.equal(stats.classes[1].detected_frames, 2);
assert.equal(stats.classes[1].instance_count, 2);

const repeatedOnly = computeEpisodeStats(scene, 0.85, 2);
assert.equal(repeatedOnly.class_count, 0);
assert.equal(repeatedOnly.instance_count, 0);
assert.equal(repeatedOnly.detected_frames, 0);
console.log('EPISODE_STATS_OK');

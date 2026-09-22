// Vocabulary shared by the views: the values mirror `taxonomy.py`, the labels
// are what a reader sees.

// Order matters: it is the order of the stacked bar and every legend.
export const STATUSES = [
  ["kept", "Kept"],
  ["partial", "Partly kept"],
  ["missed", "Missed"],
  ["quietly_dropped", "Quietly dropped"],
  ["too_early", "Too early to tell"],
  ["no_evidence", "No evidence found"],
  ["unadjudicated", "Not checked yet"],
];
export const STATUS_LABEL = Object.fromEntries(STATUSES);

export const TYPE_LABEL = {
  quantified_target: "Numeric target",
  directional_guidance: "Outlook",
  procedural_commitment: "Action pledge",
  scheduling_announcement: "Scheduled date",
};

// How firmly a promise was worded, shown as the phrase itself.
export const HEDGE_LABEL = {
  firm: "“We will”",
  intended: "“We aim to”",
  conditional: "“We expect”",
  aspirational: "“Our ambition”",
};

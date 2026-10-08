import type {
  Cultivator,
  Attributes,
  RetreatRecord,
  BreakthroughHistoryEntry,
} from '../cultivator.js';
import type { RealmType, RealmStage } from '@daoyou/constants/realms';
import type { BreakthroughModifiers } from './breakthrough-modifiers.js';

export type RetreatCultivatorFacts = Pick<
  Cultivator,
  'id' | 'name' | 'attributes' | 'realm' | 'realm_stage' | 'condition' | 'sect'
> &
  Pick<
    Cultivator,
    | 'age'
    | 'lifespan'
    | 'closed_door_years_total'
    | 'unallocated_attribute_points'
    | 'spiritual_roots'
    | 'pre_heaven_fates'
    | 'cultivation_progress'
  >;

/**
 * 闭关修炼结果
 */
export interface CultivationResult {
  cultivator: RetreatCultivatorFacts;
  summary: {
    exp_gained: number;
    exp_before: number;
    exp_after: number;
    insight_gained: number;
    epiphany_triggered: boolean;
    bottleneck_entered: boolean;
    can_breakthrough: boolean;
    progress: number; // 百分比
    /** 玩家请求的闭关年限 */
    years_requested: number;
    /** 实际消耗的年限（撞顶时按「填满当前阶段」所需缩减） */
    years_spent: number;
  };
  record: RetreatRecord;
}

/**
 * 突破尝试结果
 */
export interface BreakthroughResult {
  cultivator: RetreatCultivatorFacts;
  summary: {
    success: boolean;
    chance: number;
    roll: number;
    fromRealm: RealmType;
    fromStage: RealmStage;
    toRealm?: RealmType;
    toStage?: RealmStage;
    lifespanGained: number;
    attributeGrowth: Partial<Attributes>;
    naturalAttributeGrowth: number;
    attributePointReward: number;
    exp_progress: number;
    insight_value: number;
    exp_lost?: number;
    breakthrough_type: 'forced' | 'normal' | 'perfect';
    insight_change: number;
    inner_demon_triggered: boolean;
    modifiers: BreakthroughModifiers;
  };
  historyEntry?: BreakthroughHistoryEntry;
}
